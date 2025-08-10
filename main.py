from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        input_tokens = tokenizer.encode(text, return_tensors="pt")
        input_len = input_tokens.shape[1]

        max_len = max(10, min(100, input_len // 2))
        min_len = max(5, max_len // 2)

        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)

