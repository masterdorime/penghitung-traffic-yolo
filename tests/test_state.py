from __future__ import annotations

import copy
import pickle
import threading
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Protocol, cast

import numpy as np
import pytest

from tests.helpers import FIXED_OBSERVED_AT, FRAME_SHAPE, JAKARTA, frame_marker, make_frame
from traffic_counter.models import (
    AppSnapshot,
    AppState,
    Frame,
    FrameSlotStats,
    MinutePoint,
    PeriodName,
    PeriodSummary,
    StreamState,
)
from traffic_counter.state import Clock, LatestFrameSlot, SnapshotStore


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


def zeroed_summary(period: PeriodName) -> PeriodSummary:
    return PeriodSummary(
        period=period,
        started=False,
        total=0,
        observed_seconds=0,
        average_rate=None,
        rolling_rate=None,
    )


def snapshot_with_total(total: int, moment: datetime) -> AppSnapshot:
    base = AppSnapshot.initial(moment)
    return replace(
        base,
        periods=MappingProxyType(
            {period: replace(base.periods[period], total=total) for period in PeriodName}
        ),
    )


def minute_point(minute_start: datetime) -> MinutePoint:
    return MinutePoint(
        minute_start=minute_start,
        period=PeriodName.PAGI,
        count=3,
        observed_seconds=60,
    )


class SlotDescriptor(Protocol):
    def __get__(self, instance: object, owner: type | None = None, /) -> object: ...

    def __set__(self, instance: object, value: object, /) -> None: ...


SNAPSHOT_SLOT: Final[SlotDescriptor] = SnapshotStore.__dict__["_snapshot"]
PENDING_SLOT: Final[SlotDescriptor] = LatestFrameSlot.__dict__["_pending"]
RECEIVED_SLOT: Final[SlotDescriptor] = LatestFrameSlot.__dict__["_received"]
REPLACED_SLOT: Final[SlotDescriptor] = LatestFrameSlot.__dict__["_replaced"]


class LockExtentProbe:
    __slots__ = ("_lock", "accesses", "armed", "violations")

    def __init__(self) -> None:
        self._lock: threading.Lock | None = None
        self.accesses: list[str] = []
        self.armed = False
        self.violations: list[str] = []

    def bind(self, lock: threading.Lock) -> None:
        self._lock = lock

    def arm(self) -> None:
        self.armed = True

    def record(self, access: str) -> None:
        self.accesses.append(access)
        lock = self._lock
        if self.armed and (lock is None or not lock.locked()):
            self.violations.append(access)


class ProbedSnapshotStore(SnapshotStore):
    __slots__ = ("_probe",)

    def __init__(self, probe: LockExtentProbe, clock: Clock | None = None) -> None:
        self._probe = probe
        super().__init__(clock=clock)
        probe.bind(self._lock)
        probe.arm()

    @property
    def _snapshot(self) -> AppSnapshot:
        self._probe.record("_snapshot read")
        stored = SNAPSHOT_SLOT.__get__(self)
        assert isinstance(stored, AppSnapshot)
        return stored

    @_snapshot.setter
    def _snapshot(self, value: AppSnapshot) -> None:
        self._probe.record("_snapshot write")
        SNAPSHOT_SLOT.__set__(self, value)


class ProbedFrameSlot(LatestFrameSlot):
    __slots__ = ("_probe",)

    def __init__(self, probe: LockExtentProbe) -> None:
        self._probe = probe
        super().__init__()
        probe.bind(self._lock)
        probe.arm()

    @property
    def _pending(self) -> Frame | None:
        self._probe.record("_pending read")
        stored = PENDING_SLOT.__get__(self)
        assert isinstance(stored, Frame | None)
        return stored

    @_pending.setter
    def _pending(self, value: Frame | None) -> None:
        self._probe.record("_pending write")
        PENDING_SLOT.__set__(self, value)

    @property
    def _received(self) -> int:
        self._probe.record("_received read")
        stored = RECEIVED_SLOT.__get__(self)
        assert isinstance(stored, int)
        return stored

    @_received.setter
    def _received(self, value: int) -> None:
        self._probe.record("_received write")
        RECEIVED_SLOT.__set__(self, value)

    @property
    def _replaced(self) -> int:
        self._probe.record("_replaced read")
        stored = REPLACED_SLOT.__get__(self)
        assert isinstance(stored, int)
        return stored

    @_replaced.setter
    def _replaced(self, value: int) -> None:
        self._probe.record("_replaced write")
        REPLACED_SLOT.__set__(self, value)


def test_a_fresh_store_serves_the_initial_snapshot_without_publishing() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))

    assert store.get() == AppSnapshot.initial(at(6, 0))


def test_the_initial_snapshot_carries_every_domain_default() -> None:
    snapshot = SnapshotStore(clock=lambda: at(6, 0)).get()

    assert snapshot.generated_at == at(6, 0)
    assert snapshot.state is AppState.STOPPED
    assert snapshot.stream_state is StreamState.DISCONNECTED
    assert snapshot.actual_fps == 0.0
    assert snapshot.target_fps == 0
    assert snapshot.replaced_frames == 0
    assert snapshot.minutes == ()
    assert snapshot.csv_export_ok is False
    assert snapshot.dropped_events == 0
    assert snapshot.incomplete is False
    assert snapshot.last_error is None
    assert snapshot.tracking_session_changes == 0


def test_the_initial_snapshot_maps_every_period_to_a_zeroed_summary() -> None:
    periods = SnapshotStore(clock=lambda: at(6, 0)).get().periods

    assert set(periods) == set(PeriodName)
    assert periods == {period: zeroed_summary(period) for period in PeriodName}


def test_the_store_reads_its_injected_clock_once_for_the_initial_snapshot() -> None:
    calls: list[datetime] = []

    def clock() -> datetime:
        calls.append(at(6, 0))
        return at(6, 0)

    store = SnapshotStore(clock=clock)
    store.publish(snapshot_with_total(1, at(6, 1)))
    for _ in range(5):
        store.get()

    assert len(calls) == 1
    assert store.get().generated_at == at(6, 1)


def test_repeated_gets_before_publishing_return_the_same_object() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))

    assert store.get() is store.get()


def test_publish_makes_the_new_snapshot_readable() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    published = snapshot_with_total(7, at(6, 30))

    store.publish(published)

    assert store.get() == published
    assert store.get().periods[PeriodName.PAGI].total == 7


def test_publish_replaces_the_reference_so_the_first_object_never_changes() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(1, at(6, 1)))
    first = store.get()

    store.publish(snapshot_with_total(2, at(6, 2)))

    assert first.periods[PeriodName.PAGI].total == 1
    assert first.generated_at == at(6, 1)
    assert store.get().periods[PeriodName.PAGI].total == 2
    assert store.get() is not first


def test_publishing_never_alters_the_caller_owned_snapshot() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    published = snapshot_with_total(4, at(6, 5))

    store.publish(published)

    assert published.periods[PeriodName.PAGI].total == 4
    assert published.generated_at == at(6, 5)


def test_a_mutated_caller_mapping_does_not_reach_the_store() -> None:
    source: dict[PeriodName, PeriodSummary] = {
        period: zeroed_summary(period) for period in PeriodName
    }
    base = AppSnapshot.initial(at(6, 0))
    store = SnapshotStore(clock=lambda: at(6, 0))

    store.publish(replace(base, periods=source))
    source[PeriodName.PAGI] = replace(source[PeriodName.PAGI], total=99)
    del source[PeriodName.SIANG]

    assert set(store.get().periods) == set(PeriodName)
    assert store.get().periods[PeriodName.PAGI].total == 0


def test_a_mutated_caller_minute_sequence_does_not_reach_the_store() -> None:
    minute_list = [minute_point(at(6, 0))]
    base = AppSnapshot.initial(at(6, 0))
    store = SnapshotStore(clock=lambda: at(6, 0))

    store.publish(replace(base, minutes=cast(tuple[MinutePoint, ...], minute_list)))
    minute_list.append(minute_point(at(6, 1)))

    assert len(store.get().minutes) == 1
    assert store.get().minutes[0].minute_start == at(6, 0)


def test_publish_normalizes_a_minute_sequence_to_a_tuple() -> None:
    base = AppSnapshot.initial(at(6, 0))
    store = SnapshotStore(clock=lambda: at(6, 0))
    borrowed = [minute_point(at(6, 0))]

    store.publish(replace(base, minutes=cast(tuple[MinutePoint, ...], borrowed)))

    assert isinstance(store.get().minutes, tuple)


def test_get_never_exposes_a_plain_period_dictionary() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(1, at(6, 1)))

    periods = store.get().periods

    assert isinstance(periods, MappingProxyType)
    assert not isinstance(periods, dict)


def test_a_published_snapshot_cannot_be_pickled() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(2, at(6, 2)))

    with pytest.raises(TypeError):
        pickle.dumps(store.get())


def test_a_published_snapshot_cannot_be_deep_copied() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(2, at(6, 2)))

    with pytest.raises(TypeError):
        copy.deepcopy(store.get())


def test_the_read_only_mapping_is_what_makes_a_published_snapshot_unserializable() -> None:
    base = AppSnapshot.initial(at(6, 0))
    with_plain_mapping = replace(base, periods=dict(base.periods))
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(2, at(6, 2)))

    assert pickle.loads(pickle.dumps(with_plain_mapping)) == with_plain_mapping
    with pytest.raises(TypeError):
        pickle.dumps(store.get())


def assign_period(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    mutable[PeriodName.PAGI] = zeroed_summary(PeriodName.PAGI)
    return mapping


def delete_period(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    del mutable[PeriodName.PAGI]
    return mapping


def update_periods(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    return mutable.update({PeriodName.PAGI: zeroed_summary(PeriodName.PAGI)})


def pop_period(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    return mutable.pop(PeriodName.PAGI)


def pop_any_period(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    return mutable.popitem()


def clear_periods(mapping: Mapping[PeriodName, PeriodSummary]) -> object:
    mutable = cast("MutableMapping[PeriodName, PeriodSummary]", mapping)
    mutable.clear()
    return mapping


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        pytest.param(assign_period, TypeError, id="assign"),
        pytest.param(delete_period, TypeError, id="delete"),
        pytest.param(update_periods, AttributeError, id="update"),
        pytest.param(pop_period, AttributeError, id="pop"),
        pytest.param(pop_any_period, AttributeError, id="popitem"),
        pytest.param(clear_periods, AttributeError, id="clear"),
    ),
)
def test_the_published_period_mapping_rejects_every_mutation(
    mutation: Callable[[Mapping[PeriodName, PeriodSummary]], object],
    expected: type[BaseException],
) -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(1, at(6, 1)))

    with pytest.raises(expected):
        mutation(store.get().periods)

    assert store.get().periods[PeriodName.PAGI].total == 1


def test_a_reader_thread_only_ever_observes_published_snapshots() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    store.publish(snapshot_with_total(0, at(6, 0)))
    published_totals = 300
    done = threading.Event()
    observed: list[int] = []

    def publish_all() -> None:
        for total in range(published_totals):
            store.publish(snapshot_with_total(total, at(6, 0)))
        done.set()

    def read_all() -> None:
        while not done.is_set():
            observed.append(store.get().periods[PeriodName.PAGI].total)
        observed.append(store.get().periods[PeriodName.PAGI].total)

    writer = threading.Thread(target=publish_all)
    reader = threading.Thread(target=read_all)
    writer.start()
    reader.start()
    writer.join()
    reader.join()

    assert observed
    assert all(0 <= total < published_totals for total in observed)
    assert observed == sorted(observed)
    assert store.get().periods[PeriodName.PAGI].total == published_totals - 1


def blocks_while_the_shared_lock_is_held(
    target: LatestFrameSlot | SnapshotStore, operation: Callable[[], object]
) -> bool:
    lock = target._lock
    assert isinstance(lock, threading.Lock)
    started = threading.Event()
    finished = threading.Event()

    def run() -> None:
        started.set()
        operation()
        finished.set()

    worker = threading.Thread(target=run)
    lock.acquire()
    try:
        worker.start()
        started.wait(timeout=5.0)
        blocked = not finished.wait(timeout=0.05)
    finally:
        lock.release()
    worker.join(timeout=5.0)
    return blocked and finished.is_set()


def test_get_blocks_while_a_publisher_holds_the_store_lock() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))

    assert blocks_while_the_shared_lock_is_held(store, store.get)


def test_publish_blocks_while_a_reader_holds_the_store_lock() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))

    assert blocks_while_the_shared_lock_is_held(
        store, lambda: store.publish(snapshot_with_total(3, at(6, 3)))
    )


def test_put_blocks_while_another_thread_holds_the_slot_lock() -> None:
    slot = LatestFrameSlot()

    assert blocks_while_the_shared_lock_is_held(slot, lambda: slot.put(make_frame(1)))


def test_take_blocks_while_another_thread_holds_the_slot_lock() -> None:
    slot = LatestFrameSlot()
    slot.put(make_frame(1))

    assert blocks_while_the_shared_lock_is_held(slot, slot.take)


def test_stats_blocks_while_another_thread_holds_the_slot_lock() -> None:
    slot = LatestFrameSlot()

    assert blocks_while_the_shared_lock_is_held(slot, slot.stats)


def observed_fields(accesses: list[str]) -> set[str]:
    return {access.rsplit(" ", 1)[0] for access in accesses}


def observed_kinds(accesses: list[str]) -> set[str]:
    return {access.rsplit(" ", 1)[-1] for access in accesses}


def test_every_snapshot_field_access_happens_inside_the_store_lock() -> None:
    probe = LockExtentProbe()
    store = ProbedSnapshotStore(probe, clock=lambda: at(6, 0))

    store.publish(snapshot_with_total(4, at(6, 4)))
    store.get()

    assert observed_fields(probe.accesses) == {"_snapshot"}
    assert observed_kinds(probe.accesses) == {"read", "write"}
    assert probe.violations == []


def test_every_slot_field_access_happens_inside_the_slot_lock() -> None:
    probe = LockExtentProbe()
    slot = ProbedFrameSlot(probe)

    slot.put(make_frame(1))
    slot.put(make_frame(2))
    assert slot.take() is not None
    assert slot.stats() == FrameSlotStats(received=2, replaced=1)

    assert observed_fields(probe.accesses) == {"_pending", "_received", "_replaced"}
    assert observed_kinds(probe.accesses) == {"read", "write"}
    assert probe.violations == []


def test_the_probe_flags_an_unlocked_access_and_accepts_a_locked_one() -> None:
    probe = LockExtentProbe()
    store = ProbedSnapshotStore(probe, clock=lambda: at(6, 0))
    probe.accesses.clear()
    probe.violations.clear()

    assert isinstance(store._snapshot, AppSnapshot)
    store._snapshot = snapshot_with_total(9, at(6, 9))
    store.get()

    assert probe.violations == ["_snapshot read", "_snapshot write"]
    assert store.get().periods[PeriodName.PAGI].total == 9


def test_a_snapshot_published_from_another_thread_is_visible_to_the_next_get() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    published: list[AppSnapshot] = []

    def publish_from_thread() -> None:
        published.append(snapshot_with_total(12, at(6, 42)))
        store.publish(published[0])

    worker = threading.Thread(target=publish_from_thread)
    worker.start()
    worker.join()

    assert store.get().periods[PeriodName.PAGI].total == 12
    assert store.get() == published[0]


def test_reading_while_publishing_repeatedly_never_raises() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))
    stop = threading.Event()
    errors: list[BaseException] = []
    reads: list[int] = []

    def publish_repeatedly() -> None:
        try:
            for total in range(200):
                store.publish(snapshot_with_total(total, at(6, 0)))
        except BaseException as error:
            errors.append(error)
        finally:
            stop.set()

    def read_repeatedly() -> None:
        try:
            while not stop.is_set():
                snapshot = store.get()
                reads.append(snapshot.periods[PeriodName.PAGI].total + len(snapshot.minutes))
        except BaseException as error:
            errors.append(error)

    readers = [threading.Thread(target=read_repeatedly) for _ in range(3)]
    for reader in readers:
        reader.start()
    writer = threading.Thread(target=publish_repeatedly)
    writer.start()
    writer.join()
    for reader in readers:
        reader.join()

    assert errors == []
    assert reads


def test_latest_frame_slot_replaces_stale_frame() -> None:
    slot = LatestFrameSlot()
    first = make_frame(1)
    second = make_frame(2)
    slot.put(first)
    slot.put(second)
    assert slot.take() == second
    assert slot.stats().received == 2
    assert slot.stats().replaced == 1


def test_take_from_an_empty_slot_returns_none() -> None:
    slot = LatestFrameSlot()

    assert slot.take() is None
    assert slot.take() is None
    assert slot.stats() == FrameSlotStats(received=0, replaced=0)


def test_take_clears_the_pending_frame() -> None:
    slot = LatestFrameSlot()
    frame = make_frame(3)
    slot.put(frame)

    assert slot.take() is frame
    assert slot.take() is None


def test_a_drained_slot_does_not_record_a_replacement() -> None:
    slot = LatestFrameSlot()
    slot.put(make_frame(1))
    slot.take()
    slot.put(make_frame(2))
    slot.take()

    assert slot.stats() == FrameSlotStats(received=2, replaced=0)


def test_every_overwrite_before_a_drain_is_counted() -> None:
    slot = LatestFrameSlot()

    for marker in range(5):
        slot.put(make_frame(marker))

    assert slot.stats() == FrameSlotStats(received=5, replaced=4)


def test_stats_report_the_models_frame_slot_stats_record() -> None:
    slot = LatestFrameSlot()
    slot.put(make_frame(1))
    slot.put(make_frame(2))

    stats = slot.stats()

    assert isinstance(stats, FrameSlotStats)
    assert (stats.received, stats.replaced) == (2, 1)


def test_stats_are_cumulative_and_survive_taking() -> None:
    slot = LatestFrameSlot()
    slot.put(make_frame(1))
    slot.put(make_frame(2))
    first_reading = slot.stats()
    slot.take()

    assert slot.stats() == first_reading
    assert first_reading == FrameSlotStats(received=2, replaced=1)

    slot.put(make_frame(3))

    assert slot.stats() == FrameSlotStats(received=3, replaced=1)


def test_replaced_never_exceeds_received() -> None:
    slot = LatestFrameSlot()

    for round_index in range(10):
        slot.put(make_frame(round_index))
        if round_index % 3 == 0:
            slot.take()
        slot.put(make_frame(100 + round_index))

        assert slot.stats().replaced <= slot.stats().received


def test_the_newest_put_survives_even_though_the_timestamps_are_identical() -> None:
    slot = LatestFrameSlot()
    older = make_frame(1)
    newer = make_frame(2)
    assert older.observed_at == newer.observed_at
    slot.put(older)
    slot.put(newer)

    assert slot.take() is newer


def test_putting_the_same_frame_twice_still_records_a_replacement() -> None:
    slot = LatestFrameSlot()
    frame = make_frame(9)
    slot.put(frame)
    slot.put(frame)

    assert slot.stats() == FrameSlotStats(received=2, replaced=1)
    assert slot.take() is frame


def test_independent_slots_keep_independent_counters() -> None:
    busy = LatestFrameSlot()
    idle = LatestFrameSlot()
    busy.put(make_frame(1))
    busy.put(make_frame(2))
    busy.take()

    assert busy.stats() == FrameSlotStats(received=2, replaced=1)
    assert idle.stats() == FrameSlotStats(received=0, replaced=0)
    assert idle.take() is None


def test_a_producer_and_consumer_never_lose_a_frame_accounting() -> None:
    slot = LatestFrameSlot()
    frame_count = 400
    drained: list[int] = []
    drained_lock = threading.Lock()
    finished = threading.Event()

    def produce() -> None:
        for marker in range(frame_count):
            slot.put(make_frame(marker % 200))
        finished.set()

    def consume() -> None:
        while not finished.is_set():
            frame = slot.take()
            if frame is None:
                continue
            with drained_lock:
                drained.append(frame_marker(frame))

    producer = threading.Thread(target=produce)
    consumer = threading.Thread(target=consume)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    while (frame := slot.take()) is not None:
        with drained_lock:
            drained.append(frame_marker(frame))

    stats = slot.stats()
    assert stats.received == frame_count
    assert stats.replaced == frame_count - len(drained)
    assert stats.received == stats.replaced + len(drained)


def test_concurrent_producers_and_consumers_keep_the_replacement_ledger_exact() -> None:
    slot = LatestFrameSlot()
    producer_count = 3
    frames_per_producer = 80
    produced: Counter[int] = Counter()
    drained: Counter[int] = Counter()
    drained_lock = threading.Lock()
    producers_done = [threading.Event() for _ in range(producer_count)]

    def produce(producer_index: int) -> None:
        for offset in range(frames_per_producer):
            marker = producer_index * frames_per_producer + offset
            slot.put(make_frame(marker))
            produced[marker] += 1
        producers_done[producer_index].set()

    def consume() -> None:
        while not all(event.is_set() for event in producers_done):
            frame = slot.take()
            if frame is None:
                continue
            with drained_lock:
                drained[frame_marker(frame)] += 1

    producers = [threading.Thread(target=produce, args=(index,)) for index in range(producer_count)]
    consumers = [threading.Thread(target=consume) for _ in range(2)]
    for producer in producers:
        producer.start()
    for consumer in consumers:
        consumer.start()
    for producer in producers:
        producer.join()
    for consumer in consumers:
        consumer.join()
    while (frame := slot.take()) is not None:
        with drained_lock:
            drained[frame_marker(frame)] += 1

    stats = slot.stats()
    total_puts = producer_count * frames_per_producer
    assert sum(produced.values()) == total_puts
    assert stats.received == total_puts
    assert stats.received == stats.replaced + sum(drained.values())
    assert drained <= produced


def test_concurrent_stat_readers_never_observe_a_counter_going_backwards() -> None:
    slot = LatestFrameSlot()
    producer_count = 2
    frames_per_producer = 100
    minimum_reads = 200
    stop = threading.Event()
    readings: list[tuple[int, int]] = []
    readings_lock = threading.Lock()
    errors: list[BaseException] = []

    def produce(producer_index: int) -> None:
        try:
            for offset in range(frames_per_producer):
                slot.put(make_frame((producer_index * frames_per_producer + offset) % 200))
        except BaseException as error:
            errors.append(error)
        finally:
            stop.set()

    def read_stats() -> None:
        reads = 0
        try:
            while reads < minimum_reads or not stop.is_set():
                with readings_lock:
                    stats = slot.stats()
                    readings.append((stats.received, stats.replaced))
                reads += 1
        except BaseException as error:
            errors.append(error)

    producers = [threading.Thread(target=produce, args=(index,)) for index in range(producer_count)]
    readers = [threading.Thread(target=read_stats) for _ in range(2)]
    for producer in producers:
        producer.start()
    for reader in readers:
        reader.start()
    for producer in producers:
        producer.join()
    for reader in readers:
        reader.join()

    assert errors == []
    assert readings
    assert [received for received, _ in readings] == sorted(received for received, _ in readings)
    assert [replaced for _, replaced in readings] == sorted(replaced for _, replaced in readings)
    assert all(replaced <= received for received, replaced in readings)
    assert slot.stats().received == producer_count * frames_per_producer


def test_make_frame_builds_a_zero_filled_uint8_array() -> None:
    image = make_frame(7).image

    assert image.dtype == np.uint8
    assert image.shape == FRAME_SHAPE
    assert int(image[1, 1, 0]) == 0
    assert int(image.sum()) == 3 * 7


def test_make_frame_writes_the_marker_into_pixel_zero() -> None:
    image = make_frame(7).image

    assert [int(channel) for channel in image[0, 0]] == [7, 7, 7]
    assert frame_marker(make_frame(7)) == 7
    assert frame_marker(make_frame(0)) == 0
    assert frame_marker(make_frame(255)) == 255


def test_make_frame_uses_one_fixed_aware_jakarta_instant() -> None:
    frame = make_frame(1)

    assert frame.observed_at == FIXED_OBSERVED_AT
    assert frame.observed_at.tzinfo is JAKARTA
    assert frame.observed_at.utcoffset() == timedelta(hours=7)
    assert frame.observed_at == make_frame(2).observed_at


def test_make_frame_allocates_an_independent_array_per_call() -> None:
    first = make_frame(1)
    second = make_frame(1)

    assert first is not second
    assert first.image is not second.image
    first.image[0, 0] = 9
    assert int(second.image[0, 0, 0]) == 1


def test_make_frame_rejects_a_marker_that_is_not_a_uint8_value() -> None:
    with pytest.raises(ValueError, match="between 0 and 255"):
        make_frame(256)
    with pytest.raises(ValueError, match="between 0 and 255"):
        make_frame(-1)


@pytest.mark.parametrize("marker", (1.5, 0.0, "1", None, 3 + 0j, [1], np.uint8(1), np.int64(1)))
def test_make_frame_rejects_a_marker_that_is_not_an_integer(marker: object) -> None:
    with pytest.raises(ValueError, match="must be integers between 0 and 255"):
        make_frame(cast("int", marker))


def test_a_frame_cannot_be_hashed() -> None:
    frame = make_frame(1)

    with pytest.raises(TypeError, match="unhashable"):
        hash(frame)


def test_a_frame_cannot_be_used_as_a_dictionary_key() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        _ = {make_frame(1): "value"}


def test_a_frame_cannot_be_used_as_a_set_member() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        _ = {make_frame(1)}


def test_a_published_snapshot_is_not_hashable_despite_being_frozen() -> None:
    store = SnapshotStore(clock=lambda: at(6, 0))

    with pytest.raises(TypeError, match="unhashable"):
        hash(store.get())


def test_two_distinct_frames_cannot_be_compared_for_equality() -> None:
    first = make_frame(1)

    assert first == first
    with pytest.raises(ValueError):
        _ = first == make_frame(1)


def test_a_taken_frame_is_identified_by_identity_not_by_equality() -> None:
    slot = LatestFrameSlot()
    frame = make_frame(4)
    slot.put(frame)

    taken = slot.take()

    assert taken is frame
    assert taken is not None
    assert frame_marker(taken) == 4
