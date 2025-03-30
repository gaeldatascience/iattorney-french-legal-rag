from flask import Flask, request, render_template, jsonify, session
from chains.router import route_question
from config import SECRET_KEY

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY

@app.route("/", methods=["GET"])
def index():
    session.pop("conversation", None)
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def data():
    try:
        input_json = request.get_json()
        question = input_json.get("data", "")
        print("📩 Received question:", question)

        # Retrieve only the last exchange (if it exists)
        history_text = ""
        if "conversation" in session and len(session["conversation"]) >= 2:
            last_exchange = session["conversation"][-2:]  # Last Q&A
            history_text = "\n".join([
                f"User: {last_exchange[0]['message']}",
                f"Assistant: {last_exchange[1]['message']}"
            ])

        # Call the RAG with only the last exchange as history
        answer = route_question(question, history=history_text)

        # Overwrite the session with only the new exchange
        session["conversation"] = [
            {"role": "user", "message": question},
            {"role": "assistant", "message": answer}
        ]
        session.modified = True

        return jsonify({"response": True, "message": answer})
    except Exception as e:
        print("❌ Error in /data:", str(e))
        return jsonify({"response": False, "message": str(e)})

# Route to reset the session if needed
@app.route("/reset", methods=["GET"])
def reset():
    session.pop("conversation", None)
    return jsonify({"response": True, "message": "Conversation reset."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)