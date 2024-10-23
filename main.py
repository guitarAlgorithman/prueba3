from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import json
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env'))
port = os.getenv("PORT", 8000)  # Default to port 8000 if PORT is not set

print(f"Running on port: {port}")

app = FastAPI()
