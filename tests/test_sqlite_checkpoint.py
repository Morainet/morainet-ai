from __future__ import annotations

from morainet import Checkpoint, SQLiteCheckpointStore
from morainet.core.models import Message


async def test_sqlite_save_and_load(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
    cp = Checkpoint(trace_id="t1", query="hi", messages=[Message.user("hi")], cursor=2)
    await store.save(cp)

    loaded = await store.load("t1")
    assert loaded is not None
    assert loaded.query == "hi"
    assert loaded.cursor == 2
    assert loaded.messages[0].content == "hi"
    assert await store.load("missing") is None


async def test_sqlite_replace_and_persist(tmp_path):
    path = str(tmp_path / "ckpt.db")
    store = SQLiteCheckpointStore(path)
    await store.save(Checkpoint(trace_id="t1", query="v1"))
    await store.save(Checkpoint(trace_id="t1", query="v2"))  # overwrite
    assert (await store.load("t1")).query == "v2"
    store.close()

    # Reopen: data persisted to disk.
    store2 = SQLiteCheckpointStore(path)
    assert (await store2.load("t1")).query == "v2"
    store2.close()
