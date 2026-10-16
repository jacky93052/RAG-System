import sys
import os
import re
import json
# import pypdf
import requests
import sqlite3
import hashlib
import chromadb
from collections import defaultdict
# from transformers import pipeline
from typing import Dict, List, Any, Generator, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup

print("目前核心的 Python 路徑：")
print(sys.executable)