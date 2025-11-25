import streamlit as st
import os
import glob
import time
import json
import hashlib
import shutil
import re
import base64
from datetime import datetime
from collections import Counter
from io import BytesIO

# Library AI & Processing
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests

# OCR Check
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

# --- CONFIG & PAGE SETUP ---
st.set_page_config(
    page_title="NEURO-LINK // RAG CORE", 
    page_icon="👁️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
DEFAULT_DB_FOLDER = "faiss_index_ml_system"
DEFAULT_REGISTRY = "processed_files.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- SESSION STATE INIT ---
defaults = {
    "messages": [],
    "db": None,
    "embeddings": None,
    "registry": {},
    "query_history": [],
    "config": {
        "db_folder": DEFAULT_DB_FOLDER,
        "registry_file": DEFAULT_REGISTRY,
        "embed_model": DEFAULT_EMBED_MODEL,
        "llm_url": "http://localhost:1234/v1/chat/completions",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "temperature": 0.3,
        "max_tokens": 2048,
        "citation_mode": True,
        "system_prompt": "Kamu adalah asisten AI canggih. Jawab dalam Bahasa Indonesia dengan sitasi lengkap."
    },
    "db_stats": {"total_vectors": 0, "total_docs": 0}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- ULTRA ADVANCED VISUAL SYSTEM ---
def inject_entity_system():
    """
    CINEMATIC GRADE 3D NEURAL ENTITY SYSTEM
    - 3D Particle Matrix dengan Depth of Field
    - Multi-layered Hexagonal Grid
    - Advanced Lighting & Bloom Effects
    - Neural Network Connections
    - Chromatic Aberration
    - Holographic Scan Lines
    """
    st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');

        /* ===== GLOBAL THEME - DEEP QUANTUM SPACE ===== */
        .stApp {
            background-color: #000000;
            background-image: 
                radial-gradient(circle at 20% 30%, rgba(0, 100, 255, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(0, 255, 200, 0.03) 0%, transparent 50%),
                linear-gradient(180deg, #000000 0%, #020816 100%);
            color: #e0f7fa;
            font-family: 'Rajdhani', sans-serif;
            overflow-x: hidden;
        }

        /* TYPOGRAPHY */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            letter-spacing: 4px;
            text-transform: uppercase;
            background: linear-gradient(135deg, #ffffff 0%, #00eeff 50%, #0088ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 20px rgba(0, 200, 255, 0.5));
            animation: title-glow 3s ease-in-out infinite alternate;
        }

        @keyframes title-glow {
            0% { filter: drop-shadow(0 0 20px rgba(0, 200, 255, 0.5)); }
            100% { filter: drop-shadow(0 0 40px rgba(0, 200, 255, 0.9)); }
        }
        
        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #030712 0%, #050b1a 100%);
            border-right: 1px solid rgba(0, 255, 255, 0.15);
        }

        /* CHAT BUBBLES */
        .stChatMessage {
            background: rgba(5, 15, 30, 0.7);
            border: 1px solid rgba(0, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            margin: 10px 0;
        }
        
        div[data-testid="stChatMessage"][aria-label="user"] {
            border-left: 3px solid #ff0066;
            background: linear-gradient(90deg, rgba(255, 0, 102, 0.08) 0%, transparent 100%);
        }
        
        div[data-testid="stChatMessage"][aria-label="assistant"] {
            border-left: 3px solid #00eeff;
            background: linear-gradient(90deg, rgba(0, 238, 255, 0.08) 0%, transparent 100%);
        }

        /* ===== MASTER ENTITY CONTAINER ===== */
        .entity-master-container {
            position: relative;
            width: 100%;
            height: 600px;
            display: flex;
            justify-content: center;
            align-items: center;
            perspective: 2000px;
            margin: 20px 0;
            overflow: hidden;
            background: radial-gradient(ellipse at center, rgba(0, 50, 100, 0.1) 0%, transparent 70%);
        }

        /* VIGNETTE OVERLAY */
        .entity-master-container::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at center, transparent 30%, rgba(0, 0, 0, 0.8) 100%);
            pointer-events: none;
            z-index: 100;
        }

        /* CANVAS LAYERS */
        #particleCanvas, #hexGridCanvas, #connectionCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        #particleCanvas { z-index: 3; mix-blend-mode: screen; }
        #hexGridCanvas { z-index: 1; opacity: 0.3; }
        #connectionCanvas { z-index: 2; opacity: 0.6; }

        /* HOLOGRAPHIC SCAN LINES */
        .scan-lines-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 255, 255, 0.03) 2px,
                rgba(0, 255, 255, 0.03) 4px
            );
            pointer-events: none;
            z-index: 50;
        }

        .scan-sweep {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.8), transparent);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
            animation: scan-sweep 4s linear infinite;
            z-index: 51;
        }

        @keyframes scan-sweep {
            0% { transform: translateY(0); }
            100% { transform: translateY(600px); }
        }

        /* ENTITY CORE CONTAINER */
        .entity-core {
            position: relative;
            width: 400px;
            height: 400px;
            transform-style: preserve-3d;
            z-index: 20;
            filter: drop-shadow(0 0 60px rgba(0, 255, 255, 0.6));
        }

        /* CHROMATIC ABERRATION EFFECT */
        .entity-core::before,
        .entity-core::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            pointer-events: none;
        }

        .entity-core::before {
            background: radial-gradient(circle, rgba(255, 0, 0, 0.1) 0%, transparent 70%);
            transform: translate(-2px, -2px);
        }

        .entity-core::after {
            background: radial-gradient(circle, rgba(0, 0, 255, 0.1) 0%, transparent 70%);
            transform: translate(2px, 2px);
        }

        /* LAYER 0: EXTREME OUTER GLOW */
        .entity-field-outer {
            position: absolute;
            width: 600px;
            height: 600px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: radial-gradient(circle, transparent 30%, rgba(0, 150, 255, 0.05) 60%, transparent 100%);
            box-shadow: 
                0 0 100px rgba(0, 255, 255, 0.4),
                0 0 200px rgba(0, 200, 255, 0.3),
                0 0 300px rgba(0, 150, 255, 0.2);
            animation: pulse-field 8s ease-in-out infinite alternate;
            z-index: 1;
            filter: blur(20px);
        }

        @keyframes pulse-field {
            0% { transform: translate(-50%, -50%) scale(1); opacity: 0.5; }
            100% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
        }

        /* QUANTUM WAVE RINGS */
        .quantum-wave {
            position: absolute;
            top: 50%;
            left: 50%;
            border-radius: 50%;
            border: 2px solid rgba(0, 255, 255, 0.6);
            transform: translate(-50%, -50%);
            animation: quantum-expand 3s ease-out infinite;
            z-index: 2;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
        }

        .quantum-wave:nth-child(1) { animation-delay: 0s; }
        .quantum-wave:nth-child(2) { animation-delay: 1s; }
        .quantum-wave:nth-child(3) { animation-delay: 2s; }

        @keyframes quantum-expand {
            0% { 
                width: 50px; 
                height: 50px; 
                opacity: 1; 
                border-width: 3px; 
            }
            100% { 
                width: 500px; 
                height: 500px; 
                opacity: 0; 
                border-width: 0px; 
            }
        }

        /* MULTI-RING SYSTEM (10 Rings) */
        .entity-ring-1 { width: 390px; height: 390px; animation: spin-ring-1 60s linear infinite; }
        .entity-ring-2 { width: 360px; height: 360px; animation: spin-ring-2 50s linear infinite reverse; }
        .entity-ring-3 { width: 330px; height: 330px; animation: spin-ring-3 45s linear infinite; }
        .entity-ring-4 { width: 300px; height: 300px; animation: spin-ring-4 40s linear infinite reverse; }
        .entity-ring-5 { width: 270px; height: 270px; animation: spin-ring-5 35s linear infinite; }
        .entity-ring-6 { width: 240px; height: 240px; animation: spin-ring-6 30s linear infinite reverse; }
        .entity-ring-7 { width: 210px; height: 210px; animation: spin-ring-7 25s linear infinite; }
        .entity-ring-8 { width: 180px; height: 180px; animation: spin-ring-8 20s linear infinite reverse; }
        .entity-ring-9 { width: 150px; height: 150px; animation: spin-ring-9 15s linear infinite; }
        .entity-ring-10 { width: 120px; height: 120px; animation: spin-ring-10 10s linear infinite reverse; }

        [class^="entity-ring-"] {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            border: 1px solid rgba(0, 200, 255, 0.3);
            z-index: 5;
        }

        .entity-ring-1, .entity-ring-3, .entity-ring-5, .entity-ring-7, .entity-ring-9 {
            border-style: dashed;
            border-color: rgba(0, 220, 255, 0.4);
        }

        .entity-ring-2, .entity-ring-6, .entity-ring-10 {
            border-top: 2px solid rgba(0, 255, 255, 0.7);
            border-bottom: 2px solid rgba(0, 255, 255, 0.7);
        }

        .entity-ring-4, .entity-ring-8 {
            background: repeating-conic-gradient(
                from 0deg,
                transparent 0deg 10deg,
                rgba(0, 255, 255, 0.08) 10deg 12deg
            );
        }

        @keyframes spin-ring-1 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-2 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-3 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-4 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-5 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-6 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-7 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-8 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-9 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        @keyframes spin-ring-10 { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }

        /* SEGMENTED IRIS WITH CIRCUIT PATTERN */
        .entity-iris {
            position: absolute;
            width: 110px;
            height: 110px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: 
                radial-gradient(circle at 30% 30%, rgba(0, 100, 200, 0.6) 0%, transparent 60%),
                conic-gradient(
                    from 0deg,
                    rgba(0, 150, 255, 0.4) 0deg 45deg,
                    transparent 45deg 90deg,
                    rgba(0, 150, 255, 0.4) 90deg 135deg,
                    transparent 135deg 180deg,
                    rgba(0, 150, 255, 0.4) 180deg 225deg,
                    transparent 225deg 270deg,
                    rgba(0, 150, 255, 0.4) 270deg 315deg,
                    transparent 315deg 360deg
                );
            border: 2px solid rgba(0, 255, 255, 0.7);
            box-shadow: 
                inset 0 0 30px rgba(0, 255, 255, 0.5),
                0 0 50px rgba(0, 255, 255, 0.6);
            animation: iris-pulse 6s ease-in-out infinite alternate;
            z-index: 8;
        }

        @keyframes iris-pulse {
            0% { 
                transform: translate(-50%, -50%) scale(1) rotate(0deg); 
                box-shadow: inset 0 0 30px rgba(0, 255, 255, 0.5), 0 0 50px rgba(0, 255, 255, 0.6);
            }
            100% { 
                transform: translate(-50%, -50%) scale(1.08) rotate(5deg); 
                box-shadow: inset 0 0 40px rgba(0, 255, 255, 0.7), 0 0 70px rgba(0, 255, 255, 0.8);
            }
        }

        /* CIRCUIT PATTERN OVERLAY */
        .iris-circuit {
            position: absolute;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(0deg, transparent 48%, rgba(0, 255, 255, 0.3) 49%, rgba(0, 255, 255, 0.3) 51%, transparent 52%),
                linear-gradient(90deg, transparent 48%, rgba(0, 255, 255, 0.3) 49%, rgba(0, 255, 255, 0.3) 51%, transparent 52%);
            background-size: 10px 10px;
            border-radius: 50%;
            opacity: 0.4;
        }

        /* CORE PUPIL WITH GLITCH EFFECT */
        .entity-pupil {
            position: absolute;
            width: 50px;
            height: 50px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: 
                radial-gradient(circle at 35% 35%, 
                    #ffffff 0%, 
                    #00ffff 20%,
                    #00ccff 40%, 
                    #0099ff 60%,
                    #0066ff 80%, 
                    #000000 100%
                );
            box-shadow: 
                0 0 80px rgba(0, 255, 255, 1),
                0 0 120px rgba(0, 200, 255, 0.8),
                inset 0 0 30px rgba(0, 150, 255, 0.8);
            animation: pupil-main 4s ease-in-out infinite alternate;
            z-index: 10;
        }

        @keyframes pupil-main {
            0% { 
                box-shadow: 0 0 80px rgba(0, 255, 255, 1), 0 0 120px rgba(0, 200, 255, 0.8);
                transform: translate(-50%, -50%) scale(1);
            }
            100% { 
                box-shadow: 0 0 120px rgba(0, 255, 255, 1), 0 0 160px rgba(0, 200, 255, 0.9);
                transform: translate(-50%, -50%) scale(1.3);
            }
        }

        /* GLITCH EFFECT */
        .pupil-glitch {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: inherit;
            animation: glitch 5s infinite;
            opacity: 0;
        }

        @keyframes glitch {
            0%, 90%, 100% { opacity: 0; transform: translate(0, 0); }
            91% { opacity: 0.3; transform: translate(-2px, 2px); filter: hue-rotate(90deg); }
            92% { opacity: 0.5; transform: translate(2px, -2px); filter: hue-rotate(-90deg); }
            93% { opacity: 0; transform: translate(0, 0); }
        }

        /* HIGHLIGHT SPOT */
        .pupil-highlight {
            position: absolute;
            width: 18px;
            height: 18px;
            top: 20%;
            left: 25%;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 255, 255, 1) 0%, transparent 70%);
            filter: blur(3px);
            z-index: 11;
        }

        /* ===== THINKING STATE ===== */
        .state-thinking .entity-pupil {
            background: radial-gradient(circle at 35% 35%, #ffffff 0%, #ff3300 30%, #ff0066 60%, #000000 100%);
            box-shadow: 0 0 100px rgba(255, 50, 0, 1), 0 0 150px rgba(255, 0, 100, 0.8);
            animation: pupil-thinking 0.3s ease-in-out infinite alternate;
        }

        @keyframes pupil-thinking {
            0% { transform: translate(-50%, -50%) scale(1); }
            100% { transform: translate(-50%, -50%) scale(1.4); }
        }

        .state-thinking .entity-iris {
            border-color: rgba(255, 50, 0, 0.9);
            animation: iris-thinking 0.8s linear infinite;
            box-shadow: inset 0 0 40px rgba(255, 50, 0, 0.7), 0 0 80px rgba(255, 0, 100, 0.9);
        }

        @keyframes iris-thinking {
            0% { transform: translate(-50%, -50%) rotate(0deg) scale(1); }
            100% { transform: translate(-50%, -50%) rotate(360deg) scale(1.15); }
        }

        /* STATUS TEXT */
        .entity-status {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'Orbitron', sans-serif;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 6px;
            color: #00ffff;
            text-shadow: 0 0 15px rgba(0, 255, 255, 1), 0 0 30px rgba(0, 200, 255, 0.8);
            text-align: center;
            white-space: nowrap;
            animation: status-flicker 3s ease-in-out infinite;
            z-index: 101;
        }

        @keyframes status-flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .state-thinking .entity-status {
            color: #ff3300;
            text-shadow: 0 0 20px rgba(255, 50, 0, 1), 0 0 40px rgba(255, 0, 100, 0.9);
            animation: status-pulse 0.5s ease-in-out infinite;
        }

        @keyframes status-pulse {
            0%, 100% { opacity: 1; transform: translateX(-50%) scale(1); }
            50% { opacity: 0.7; transform: translateX(-50%) scale(1.05); }
        }
    </style>
    """, unsafe_allow_html=True)

def render_entity_advanced(state="idle"):
    """
    ULTRA CINEMATIC ENTITY RENDERER
    Complete dengan semua enhancement
    """
    state_class = f"state-{state}"
    
    if state == "thinking":
        status_text = "◈ NEURAL PROCESSING // QUANTUM ANALYSIS ACTIVE ◈"
    elif state == "scanning":
        status_text = "◈ DEEP SCAN INITIATED // VECTOR SEARCH ◈"
    elif state == "error":
        status_text = "◈ SYSTEM ANOMALY DETECTED // RECALIBRATING ◈"
    else:
        status_text = "◈ NEURAL LINK ESTABLISHED // STANDBY MODE ◈"
    
    html_content = f"""
<div class="entity-master-container">
<canvas id="hexGridCanvas"></canvas>
<canvas id="connectionCanvas"></canvas>
<canvas id="particleCanvas"></canvas>

<div class="scan-lines-overlay"></div>
<div class="scan-sweep"></div>

<div class="quantum-wave"></div>
<div class="quantum-wave"></div>
<div class="quantum-wave"></div>

<div class="entity-core {state_class}" id="entityCore">
<div class="entity-field-outer"></div>

<div class="entity-ring-1"></div>
<div class="entity-ring-2"></div>
<div class="entity-ring-3"></div>
<div class="entity-ring-4"></div>
<div class="entity-ring-5"></div>
<div class="entity-ring-6"></div>
<div class="entity-ring-7"></div>
<div class="entity-ring-8"></div>
<div class="entity-ring-9"></div>
<div class="entity-ring-10"></div>

<div class="entity-iris">
<div class="iris-circuit"></div>
</div>

<div class="entity-pupil" id="entityPupil">
<div class="pupil-glitch"></div>
<div class="pupil-highlight"></div>
</div>
</div>

<div class="entity-status">{status_text}</div>
</div>

<script>
(function() {{
const particleCanvas = document.getElementById('particleCanvas');
const hexCanvas = document.getElementById('hexGridCanvas');
const connCanvas = document.getElementById('connectionCanvas');
if (!particleCanvas || !hexCanvas || !connCanvas) return;

const pCtx = particleCanvas.getContext('2d');
const hCtx = hexCanvas.getContext('2d');
const cCtx = connCanvas.getContext('2d');

const width = particleCanvas.offsetWidth;
const height = particleCanvas.offsetHeight;
particleCanvas.width = hexCanvas.width = connCanvas.width = width;
particleCanvas.height = hexCanvas.height = connCanvas.height = height;

const centerX = width / 2;
const centerY = height / 2;

// ===== HEXAGONAL GRID =====
function drawHexGrid() {{
const hexSize = 30;
const cols = Math.ceil(width / (hexSize * 1.5)) + 2;
const rows = Math.ceil(height / (hexSize * Math.sqrt(3))) + 2;
hCtx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
hCtx.lineWidth = 0.5;
for (let row = -1; row < rows; row++) {{
for (let col = -1; col < cols; col++) {{
const x = col * hexSize * 1.5;
const y = row * hexSize * Math.sqrt(3) + (col % 2) * (hexSize * Math.sqrt(3) / 2);
drawHexagon(hCtx, x, y, hexSize);
}}
}}
}}

function drawHexagon(ctx, x, y, size) {{
ctx.beginPath();
for (let i = 0; i < 6; i++) {{
const angle = (Math.PI / 3) * i;
const hx = x + size * Math.cos(angle);
const hy = y + size * Math.sin(angle);
if (i === 0) ctx.moveTo(hx, hy);
else ctx.lineTo(hx, hy);
}}
ctx.closePath();
ctx.stroke();
}}

drawHexGrid();

// ===== 3D PARTICLE SYSTEM WITH BOKEH =====
const particles = [];
const particleCount = 300;

class Particle {{
constructor() {{ this.reset(); }}
reset() {{
const angle = Math.random() * Math.PI * 2;
const distance = Math.random() * 250 + 50;
this.x = centerX + Math.cos(angle) * distance;
this.y = centerY + Math.sin(angle) * distance;
this.z = Math.random() * 500 - 250;
const dir = Math.atan2(this.y - centerY, this.x - centerX);
this.vx = Math.cos(dir) * 0.3;
this.vy = Math.sin(dir) * 0.3;
this.vz = Math.random() * 0.5 - 0.25;
this.baseSize = Math.random() * 12 + 1;
this.opacity = Math.random() * 0.6 + 0.3;
this.life = Math.random() * 300 + 200;
this.maxLife = this.life;
this.hue = Math.random() * 20 + 180;
}}
update() {{
this.x += this.vx;
this.y += this.vy;
this.z += this.vz;
this.life--;
if (this.life <= 0 || this.z > 250) this.reset();
}}
draw() {{
const lifeFactor = this.life / this.maxLife;
const scale = 1 + (this.z / 500);
const size = this.baseSize * scale;
const opacity = this.opacity * lifeFactor * Math.max(0, 1 - Math.abs(this.z) / 250);
const blur = Math.abs(this.z) / 50 + 2;
pCtx.shadowBlur = blur * 3;
pCtx.shadowColor = `hsla(${{this.hue}}, 100%, 50%, ${{opacity}})`;
pCtx.fillStyle = `hsla(${{this.hue}}, 100%, 70%, ${{opacity}})`;
pCtx.beginPath();
pCtx.arc(this.x, this.y, size, 0, Math.PI * 2);
pCtx.fill();
pCtx.shadowBlur = 0;
}}
}}

for (let i = 0; i < particleCount; i++) particles.push(new Particle());

// Sort by Z for depth
particles.sort((a, b) => a.z - b.z);

// ===== NEURAL CONNECTION LINES =====
function drawConnections() {{
cCtx.clearRect(0, 0, width, height);
cCtx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
cCtx.lineWidth = 0.5;
const maxDist = 80;
for (let i = 0; i < particles.length; i++) {{
for (let j = i + 1; j < particles.length; j++) {{
const dx = particles[i].x - particles[j].x;
const dy = particles[i].y - particles[j].y;
const dist = Math.sqrt(dx * dx + dy * dy);
if (dist < maxDist) {{
const opacity = (1 - dist / maxDist) * 0.3;
cCtx.strokeStyle = `rgba(0, 255, 255, ${{opacity}})`;
cCtx.beginPath();
cCtx.moveTo(particles[i].x, particles[i].y);
cCtx.lineTo(particles[j].x, particles[j].y);
cCtx.stroke();
}}
}}
}}
}}

let connFrame = 0;
function animate() {{
pCtx.clearRect(0, 0, width, height);
particles.forEach(p => {{ p.update(); p.draw(); }});
connFrame++;
if (connFrame % 3 === 0) drawConnections();
requestAnimationFrame(animate);
}}
animate();

// ===== PUPIL MOUSE TRACKING =====
const pupil = document.getElementById('entityPupil');
const entityCore = document.getElementById('entityCore');
if (pupil && entityCore) {{
document.addEventListener('mousemove', (e) => {{
const rect = entityCore.getBoundingClientRect();
const cx = rect.left + rect.width / 2;
const cy = rect.top + rect.height / 2;
const dx = e.clientX - cx;
const dy = e.clientY - cy;
const dist = Math.sqrt(dx*dx + dy*dy);
const max = 20;
const moveX = (dx/dist) * Math.min(dist/8, max);
const moveY = (dy/dist) * Math.min(dist/8, max);
pupil.style.transform = `translate(calc(-50% + ${{moveX}}px), calc(-50% + ${{moveY}}px))`;
}});
}}

// ===== DATA STREAM VERTICAL (Matrix Rain) =====
const streamCanvas = document.createElement('canvas');
streamCanvas.style.position = 'absolute';
streamCanvas.style.top = '0';
streamCanvas.style.left = '0';
streamCanvas.style.width = '100%';
streamCanvas.style.height = '100%';
streamCanvas.style.pointerEvents = 'none';
streamCanvas.style.zIndex = '4';
streamCanvas.style.opacity = '0.3';
streamCanvas.width = width;
streamCanvas.height = height;
document.querySelector('.entity-master-container').appendChild(streamCanvas);

const sCtx = streamCanvas.getContext('2d');
const fontSize = 14;
const columns = Math.floor(width / fontSize);
const drops = Array(columns).fill(1);
const chars = '01アイウエオカキクケコサシスセソ';

function drawMatrix() {{
sCtx.fillStyle = 'rgba(0, 0, 0, 0.05)';
sCtx.fillRect(0, 0, width, height);
sCtx.fillStyle = '#00ffff';
sCtx.font = fontSize + 'px monospace';
for (let i = 0; i < drops.length; i++) {{
const text = chars.charAt(Math.floor(Math.random() * chars.length));
const x = i * fontSize;
const y = drops[i] * fontSize;
sCtx.fillText(text, x, y);
if (y > height && Math.random() > 0.975) drops[i] = 0;
drops[i]++;
}}
}}

setInterval(drawMatrix, 50);
}})();
</script>
"""
    
    st.markdown(html_content, unsafe_allow_html=True)

# --- BACKEND LOGIC ---
def hash_file(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()

def hash_bytes(data):
    return hashlib.md5(data).hexdigest()

def load_registry():
    rf = st.session_state.config["registry_file"]
    if os.path.exists(rf):
        with open(rf, 'r') as f: st.session_state.registry = json.load(f)

def save_registry():
    rf = st.session_state.config["registry_file"]
    with open(rf, 'w') as f: json.dump(st.session_state.registry, f, indent=2)

def extract_pdf_with_pages(file_path=None, file_bytes=None, filename="uploaded"):
    pages_data = [] 
    ocr_used = False
    try:
        pdf = pdfplumber.open(file_path if file_path else BytesIO(file_bytes))
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            method = "TEXT"
            if len(text.strip()) < 10 and OCR_AVAILABLE:
                try:
                    ocr = RapidOCR()
                    im = page.to_image(resolution=200)
                    img_bytes = BytesIO(); im.original.save(img_bytes, format='PNG')
                    res, _ = ocr(img_bytes.getvalue())
                    if res:
                        text = " ".join([r[1] for r in res])
                        method = "OCR"
                        ocr_used = True
                except: pass
            if text.strip():
                pages_data.append({"page_num": i+1, "text": text, "method": method})
        pdf.close()
        return pages_data, total_pages, ocr_used, None
    except Exception as e:
        return [], 0, False, str(e)

@st.cache_resource
def load_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_database():
    cfg = st.session_state.config
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings(cfg["embed_model"])
    if os.path.exists(cfg["db_folder"]):
        try:
            st.session_state.db = FAISS.load_local(cfg["db_folder"], st.session_state.embeddings, allow_dangerous_deserialization=True)
            st.session_state.db_stats["total_vectors"] = st.session_state.db.index.ntotal
            return True
        except: pass
    return False

def save_database():
    if st.session_state.db:
        st.session_state.db.save_local(st.session_state.config["db_folder"])
        return True
    return False

def process_documents(files_data):
    cfg = st.session_state.config
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings(cfg["embed_model"])
    
    all_docs = []
    results = []
    
    for item in files_data:
        is_tuple = isinstance(item, tuple)
        fname = item[0] if is_tuple else os.path.basename(item)
        fhash = hash_bytes(item[1]) if is_tuple else hash_file(item)
        
        if fhash in st.session_state.registry:
            results.append({"file": fname, "status": "SKIPPED"})
            continue
            
        pages, total, ocr, err = extract_pdf_with_pages(file_bytes=item[1], filename=fname) if is_tuple else extract_pdf_with_pages(file_path=item)
        
        if err or not pages:
            results.append({"file": fname, "status": "FAILED", "reason": err})
            continue
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
        file_chunks = []
        for p in pages:
            doc = Document(page_content=p["text"], metadata={"source": fname, "page": p["page_num"], "total_pages": total, "method": p["method"]})
            file_chunks.extend(splitter.split_documents([doc]))
            
        all_docs.extend(file_chunks)
        st.session_state.registry[fhash] = {"file": fname, "chunks": len(file_chunks), "pages": total, "ocr": ocr, "time": datetime.now().isoformat()}
        results.append({"file": fname, "status": "OK"})
    
    if all_docs:
        if st.session_state.db: st.session_state.db.add_documents(all_docs)
        else: st.session_state.db = FAISS.from_documents(all_docs, st.session_state.embeddings)
        save_database(); save_registry()
        st.session_state.db_stats["total_vectors"] = st.session_state.db.index.ntotal
    
    return results

def search_and_answer(query):
    if not st.session_state.db: return None, [], "DB OFFLINE"
    results = st.session_state.db.similarity_search_with_score(query, k=st.session_state.config["top_k"])
    
    context, sources = [], []
    for doc, score in results:
        s_score = max(0.0, min(float(score), 1.0))
        meta = doc.metadata
        sources.append({"source": meta.get("source"), "page": meta.get("page"), "score": 1-s_score, "preview": doc.page_content[:200]})
        context.append(f"[Dokumen: {meta.get('source')}, Hal: {meta.get('page')}]\n{doc.page_content}")
    
    messages = [
        {"role": "system", "content": st.session_state.config["system_prompt"]},
        {"role": "user", "content": f"KONTEKS:\n{''.join(context)}\n\nPERTANYAAN: {query}"}
    ]
    return messages, sources, None

def query_llm_stream(messages):
    cfg = st.session_state.config
    try:
        resp = requests.post(
            cfg["llm_url"],
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "local-model", "messages": messages, "temperature": cfg["temperature"], "stream": True}),
            stream=True, timeout=120
        )
        if resp.status_code == 200: return resp
    except: pass
    raise Exception("LLM Connection Failed")

# --- UI PAGES ---
def render_sidebar():
    with st.sidebar:
        st.title("🔮 NEURO-LINK")
        st.caption("Secure Intelligence Interface")
        
        db_ok = st.session_state.db is not None
        st.markdown(f"""
        <div style='border:1px solid #00ffff33; padding:15px; border-radius:8px; background:rgba(0,0,0,0.5); margin-bottom:20px;'>
            <div style='color:#888; font-size:10px; letter-spacing:2px; margin-bottom:5px;'>SYSTEM DIAGNOSTICS</div>
            <div style='display:flex; justify-content:space-between;'>
                <span>CORE:</span> <span style='color:{'#00ff00' if db_ok else '#ff0000'}'>{'● ONLINE' if db_ok else '● OFFLINE'}</span>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span>VECTORS:</span> <span style='color:#00ffff'>{st.session_state.db_stats['total_vectors']:,}</span>
            </div>
             <div style='display:flex; justify-content:space-between;'>
                <span>DOCS:</span> <span style='color:#00ffff'>{len(st.session_state.registry)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio("NAVIGATION", [
            "💬 NEURO CHAT", "🔍 DEEP SEARCH", "📤 UPLINK DATA", "📂 SECTOR SCAN", 
            "🗄️ CORE DATABASE", "📊 ANALYTICS", "⚙️ PROTOCOLS"
        ], label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 REBOOT"):
                load_database(); load_registry(); st.rerun()
        with col2:
            if st.button("🗑️ PURGE"):
                st.session_state.messages = []; st.rerun()
        return page

def page_chat():
    inject_entity_system()
    st.markdown("## <span style='color:#00ffff'>//</span> INTERFACE COMM LINK", unsafe_allow_html=True)
    
    entity_ph = st.empty()
    with entity_ph.container(): render_entity_advanced("idle")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("📂 ENCRYPTED SOURCE PACKETS"):
                        for s in msg["sources"]:
                            st.markdown(f"**> {s['source']}** (Pg. {s['page']})")
                            st.progress(s['score'])
                            st.caption(s['preview'])

    if prompt := st.chat_input("Transmit query to core..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
        
        with entity_ph.container(): render_entity_advanced("thinking")
        
        with chat_container:
            with st.chat_message("assistant"):
                resp_ph = st.empty()
                msgs, srcs, err = search_and_answer(prompt)
                
                if err:
                    resp_ph.error(err)
                else:
                    full_resp = ""
                    try:
                        resp = query_llm_stream(msgs)
                        for line in resp.iter_lines():
                            if line:
                                dec = line.decode('utf-8').strip()
                                if dec.startswith("data: ") and dec != "data: [DONE]":
                                    try:
                                        chunk = json.loads(dec[6:])['choices'][0]['delta'].get('content', '')
                                        full_resp += chunk
                                        resp_ph.markdown(full_resp + "█")
                                    except: pass
                        resp_ph.markdown(full_resp)
                        st.session_state.messages.append({"role": "assistant", "content": full_resp, "sources": srcs})
                        if srcs:
                            with st.expander("📂 DATA FRAGMENTS"):
                                for s in srcs:
                                    st.markdown(f"**{s['source']}** (Pg. {s['page']})")
                                    st.progress(s['score'])
                    except Exception as e:
                        resp_ph.error(f"LINK FAILURE: {e}")
        
        with entity_ph.container(): render_entity_advanced("idle")

def page_semantic_search():
    inject_entity_system()
    st.header("🔍 DEEP SEARCH MATRIX")
    if not st.session_state.db: return st.warning("DB OFFLINE")
    
    query = st.text_input("SEARCH VECTOR")
    if query:
        res = st.session_state.db.similarity_search_with_score(query, k=5)
        for i, (doc, score) in enumerate(res):
            with st.expander(f"RESULT #{i+1} | {doc.metadata.get('source')} | CONF: {1-float(score):.2f}"):
                st.markdown(doc.page_content)

def page_upload():
    inject_entity_system()
    st.header("📤 UPLINK DATA")
    uploaded = st.file_uploader("SELECT PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("INITIATE UPLINK"):
        process_documents([(f.name, f.read()) for f in uploaded])
        st.success("DATA INGESTED")

def page_scan():
    inject_entity_system()
    st.header("📂 SECTOR SCAN")
    path = st.text_input("PATH", value=r"D:\PAPER AI")
    if st.button("SCAN"):
        files = glob.glob(os.path.join(path, "*.pdf"))
        st.info(f"FOUND {len(files)} FILES")
        if st.button("PROCESS TARGETS"):
            process_documents(files)
            st.success("DONE")

def page_database():
    inject_entity_system()
    st.header("🗄️ CORE STORAGE")
    st.json(st.session_state.db_stats)
    st.dataframe([v for v in st.session_state.registry.values() if isinstance(v, dict)])
    if st.button("NUKE DB"):
        shutil.rmtree(st.session_state.config["db_folder"], ignore_errors=True)
        st.session_state.db = None
        st.rerun()

def page_settings():
    inject_entity_system()
    st.header("⚙️ PROTOCOLS")
    cfg = st.session_state.config
    cfg["llm_url"] = st.text_input("LLM URL", cfg["llm_url"])
    if st.button("SAVE"): st.success("SAVED")

def page_analytics():
    inject_entity_system()
    st.header("📊 ANALYTICS DASHBOARD")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TOTAL DOCUMENTS", len(st.session_state.registry))
    with col2:
        st.metric("VECTOR EMBEDDINGS", st.session_state.db_stats['total_vectors'])
    with col3:
        total_chunks = sum(v.get('chunks', 0) for v in st.session_state.registry.values() if isinstance(v, dict))
        st.metric("TOTAL CHUNKS", total_chunks)
    
    if st.session_state.registry:
        st.subheader("Document Processing History")
        docs_list = []
        for fhash, data in st.session_state.registry.items():
            if isinstance(data, dict):
                docs_list.append({
                    "File": data.get("file", "Unknown"),
                    "Pages": data.get("pages", 0),
                    "Chunks": data.get("chunks", 0),
                    "OCR Used": "✓" if data.get("ocr", False) else "✗",
                    "Processed": data.get("time", "N/A")
                })
        st.dataframe(docs_list, use_container_width=True)

# --- MAIN ---
def main():
    load_registry()
    if st.session_state.db is None: load_database()
    
    page = render_sidebar()
    if page == "💬 NEURO CHAT": page_chat()
    elif page == "🔍 DEEP SEARCH": page_semantic_search()
    elif page == "📤 UPLINK DATA": page_upload()
    elif page == "📂 SECTOR SCAN": page_scan()
    elif page == "🗄️ CORE DATABASE": page_database()
    elif page == "⚙️ PROTOCOLS": page_settings()
    elif page == "📊 ANALYTICS": page_analytics()

if __name__ == "__main__":
    main()