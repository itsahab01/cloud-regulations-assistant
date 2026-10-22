#!/usr/bin/env python3
"""
Web Application Server for CST Cloud Computing Regulatory Framework (RS10) RAG System
Author: SDAIA Agentic AI Program
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from src.rag_engine import CloudRegulationsRAG
from src.storage_rules import StorageRulesEngine

app = Flask(__name__)
CORS(app)

# Initialize RAG engine
rag_engine = CloudRegulationsRAG()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query_api():
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "").strip()
        custom_key = data.get("api_key", "").strip()

        if custom_key:
            rag_engine.update_api_key(custom_key)

        if not user_query:
            return jsonify({"error": "الرجاء كتابة نوع البيانات أو الاستفسار."}), 400

        result = rag_engine.query(user_query)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Returns the regulatory classification summary table."""
    return jsonify({
        "status": "success",
        "classifications": StorageRulesEngine.CLASSIFICATIONS
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "CST Cloud Regulations RAG Engine",
        "document": "CST RS10 - 4th Edition (Oct 2023)"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting CST Cloud RAG Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
