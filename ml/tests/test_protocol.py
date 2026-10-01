from sellanything_ml.server import PROTOCOL_VERSION, handle_message


def test_ping():
    assert handle_message({"type": "ping"}) == {"type": "pong", "protocol": PROTOCOL_VERSION}


def test_classify_returns_top_k():
    reply = handle_message({"type": "classify", "strokes": [[[0, 10], [0, 10]]]})
    assert reply["type"] == "classify_result"
    assert reply["top_k"][0]["label"]


def test_classify_rejects_empty_strokes():
    assert handle_message({"type": "classify", "strokes": []})["type"] == "error"


def test_chat_reply_shape():
    reply = handle_message({"type": "chat", "session": "s1", "text": "I want to sell a bike"})
    assert reply["type"] == "chat_reply"
    assert "offer" in reply


def test_unknown_type():
    assert handle_message({"type": "nope"})["type"] == "error"
