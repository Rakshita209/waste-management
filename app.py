from flask import Flask, render_template, request, jsonify, session
import sqlite3
import json
import requests
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = "waste-management-chatbot-secret-key"

DATABASE = "db.sqlite"
DATA_FILE = "data.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "ollama3.2:latest"


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def load_waste_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_message(session_id, role, message):
    conn = get_db()

    conn.execute(
        """
        INSERT INTO conversations
        (session_id, role, message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            message,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_memory(session_id, limit=12):
    conn = get_db()

    rows = conn.execute(
        """
        SELECT role, message
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit)
    ).fetchall()

    conn.close()

    rows = list(reversed(rows))

    return [
        {
            "role": row["role"],
            "message": row["message"]
        }
        for row in rows
    ]


def search_waste_information(user_message):
    waste_data = load_waste_data()

    message = user_message.lower()

    matches = []

    for item in waste_data:
        searchable_text = " ".join([
            str(item.get("name", "")),
            str(item.get("description", "")),
            str(item.get("type", "")),
            str(item.get("recycle_details", "")),
            str(item.get("recommendation", "")),
            str(item.get("suggestion", ""))
        ]).lower()

        words = [
            word.strip(".,!?;:")
            for word in message.split()
            if len(word.strip(".,!?;:")) > 2
        ]

        score = sum(1 for word in words if word in searchable_text)

        if score > 0:
            matches.append((score, item))

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [item[1] for item in matches[:5]]


def build_prompt(user_message, memory, waste_information):
    memory_text = "\n".join(
        f"{item['role']}: {item['message']}"
        for item in memory
    )

    waste_text = json.dumps(
        waste_information,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are EcoBot, a helpful waste-management assistant.

Your job is to help users identify waste and dispose of it responsibly.

You should provide:
1. Waste description
2. Waste type/category
3. Recycling details
4. Recommendation
5. Practical suggestion

Use the provided waste-management knowledge when relevant.

Rules:
- Do not invent recycling rules if the supplied knowledge contains the answer.
- If you are uncertain, clearly say so.
- Encourage recycling, reuse, composting, and responsible disposal.
- Do not recommend burning waste.
- Keep responses clear and practical.
- Remember relevant information from the conversation history.
- If the user previously mentioned a waste item, use that context when answering follow-up questions.

WASTE MANAGEMENT KNOWLEDGE:
{waste_text}

CONVERSATION MEMORY:
{memory_text}

USER:
{user_message}

Respond as a friendly waste-management assistant.
"""


def ask_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "response",
        "Sorry, I could not generate a response."
    ).strip()


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid request."
            }), 400

        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({
                "error": "Please enter a message."
            }), 400

        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())

        session_id = session["session_id"]

        save_message(
            session_id,
            "user",
            user_message
        )

        memory = get_memory(session_id)

        waste_information = search_waste_information(
            user_message
        )

        prompt = build_prompt(
            user_message,
            memory,
            waste_information
        )

        bot_response = ask_ollama(prompt)

        save_message(
            session_id,
            "assistant",
            bot_response
        )

        return jsonify({
            "response": bot_response
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "Cannot connect to Ollama. "
                "Make sure Ollama is running on "
                "http://localhost:11434."
            )
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Ollama took too long to respond."
        }), 504

    except requests.exceptions.RequestException as error:
        return jsonify({
            "error": f"Ollama error: {str(error)}"
        }), 500

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/history", methods=["GET"])
def history():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    messages = get_memory(
        session["session_id"],
        limit=50
    )

    return jsonify({
        "messages": messages
    })


@app.route("/api/clear", methods=["POST"])
def clear_memory():
    if "session_id" not in session:
        return jsonify({
            "success": True
        })

    conn = get_db()

    conn.execute(
        """
        DELETE FROM conversations
        WHERE session_id = ?
        """,
        (session["session_id"],)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "success": True
    })


@app.route("/api/waste", methods=["GET"])
def waste_list():
    return jsonify(load_waste_data())


initialize_database()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )