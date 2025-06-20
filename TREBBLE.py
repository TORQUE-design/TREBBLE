import os
import sys
import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import queue

# Gemini
import google.generativeai as genai
from google.generativeai import types

# PDF Support
try:
    from fpdf import FPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------- CONFIGURATION ----------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change if needed
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBw2fK-rDo7MuOGBCqSDqDzoHGLxTFLo1I")
GEMINI_MODEL = "gemini-1.5-flash"

# ---------- GEMINI CLIENT ----------
class GeminiClient:
    def __init__(self, api_key=GEMINI_API_KEY, model=GEMINI_MODEL):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model,
            generation_config=types.GenerationConfig(
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024
            ),
            safety_settings={
                'HATE': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
        )
    def explain(self, text):
        try:
            prompt = f"Explain this text in simple terms:\n{text[:500]}"
            response = self.client.generate_content(prompt, request_options={'timeout': 10})
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

# ---------- CAMERA DETECTION ----------
def list_available_cameras(max_check=5):
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available or ['0']

# ---------- OCR FUNCTION ----------
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_text(frame):
    try:
        proc = preprocess_image(frame)
        config = '--psm 6 --oem 1'
        text = pytesseract.image_to_string(proc, config=config).strip()
        return text
    except Exception as e:
        return f"OCR Error: {str(e)}"

# ---------- WORKER THREADS ----------
def ocr_worker(frame_queue, ocr_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        text = extract_text(frame)
        if text and len(text) > 5 and not text.startswith("OCR Error"):
            try:
                if ocr_queue.full():
                    ocr_queue.get_nowait()
                ocr_queue.put_nowait(text)
            except:
                pass

def ai_worker(ai_task_queue, ai_result_queue, gemini_client):
    while True:
        text = ai_task_queue.get()
        if text is None:
            break
        explanation = gemini_client.explain(text)
        try:
            ai_result_queue.put_nowait(explanation)
        except queue.Full:
            pass

# ---------- MAIN APPLICATION ----------
class SmartBookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Book Assistant (Gemini Edition)")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Camera selection
        self.cameras = list_available_cameras()
        self.current_cam = int(self.cameras[0])
        self.cap = None
        self.running = True
        self.paused = False
        self.frame_count = 0

        # Queues and Threads
        self.frame_queue = queue.Queue(maxsize=1)
        self.ocr_queue = queue.Queue(maxsize=1)
        self.ai_task_queue = queue.Queue(maxsize=1)
        self.ai_result_queue = queue.Queue(maxsize=1)
        self.gemini_client = GeminiClient()

        # GUI
        self.create_widgets()
        self.start_camera()

        # Start workers
        threading.Thread(target=ocr_worker, args=(self.frame_queue, self.ocr_queue), daemon=True).start()
        threading.Thread(target=ai_worker, args=(self.ai_task_queue, self.ai_result_queue, self.gemini_client), daemon=True).start()

        # Start loops
        self.after(100, self.update_camera)
        self.after(1000, self.update_texts)

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Camera panel
        left = ttk.LabelFrame(main_frame, text="Camera")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.preview = ttk.Label(left)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=3)

        # Camera dropdown
        ttk.Label(btns, text="Camera:").pack(side=tk.LEFT)
        self.cam_combo = ttk.Combobox(btns, values=self.cameras, state='readonly', width=5)
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.LEFT, padx=3)
        self.cam_combo.bind("<<ComboboxSelected>>", self.change_camera)

        ttk.Button(btns, text="Pause/Resume", command=self.toggle_pause).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Snapshot", command=self.snapshot).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Save Notes", command=self.save_notes).pack(side=tk.RIGHT, padx=3)

        # Text panels
        right = ttk.Frame(main_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ocr_frame = ttk.LabelFrame(right, text="Detected Text")
        ocr_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ocr_text = tk.Text(ocr_frame, wrap=tk.WORD, height=15, font=("Arial", 10))
        self.ocr_text.pack(fill=tk.BOTH, expand=True)
        ai_frame = ttk.LabelFrame(right, text="Gemini Explanation")
        ai_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ai_text = tk.Text(ai_frame, wrap=tk.WORD, height=10, font=("Arial", 10))
        self.ai_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def start_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.current_cam, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def change_camera(self, event):
        new_cam = int(self.cam_combo.get())
        if new_cam != self.current_cam:
            self.current_cam = new_cam
            self.start_camera()
            self.status_var.set(f"Switched to camera {self.current_cam}")

    def update_camera(self):
        if self.running and self.cap and not self.paused:
            ret, frame = self.cap.read()
            if ret:
                # Only process every 20th frame for OCR
                if self.frame_count % 20 == 0:
                    try:
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame.copy())
                    except:
                        pass
                self.frame_count += 1
                # Display
                frame_disp = cv2.resize(frame, (640, 480))
                cv2image = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview.imgtk = imgtk
                self.preview.configure(image=imgtk)
        self.after(50, self.update_camera)

    def update_texts(self):
        try:
            # OCR
            if not self.ocr_queue.empty():
                text = self.ocr_queue.get_nowait()
                self.ocr_text.delete("1.0", tk.END)
                self.ocr_text.insert(tk.END, text)
                # Queue for AI
                if self.ai_task_queue.full():
                    self.ai_task_queue.get_nowait()
                self.ai_task_queue.put_nowait(text[:500])
            # AI
            if not self.ai_result_queue.empty():
                explanation = self.ai_result_queue.get_nowait()
                self.ai_text.delete("1.0", tk.END)
                self.ai_text.insert(tk.END, explanation)
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        self.after(1000, self.update_texts)

    def toggle_pause(self):
        self.paused = not self.paused
        self.status_var.set("Paused" if self.paused else "Running")

    def snapshot(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                text = extract_text(frame)
                self.ocr_text.delete("1.0", tk.END)
                self.ocr_text.insert(tk.END, text)
                if self.ai_task_queue.full():
                    self.ai_task_queue.get_nowait()
                self.ai_task_queue.put_nowait(text[:500])
                self.status_var.set("Snapshot processed")

    def save_notes(self):
        text = self.ocr_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "No text to save!")
            return

        filetypes = []
        if PDF_SUPPORT:
            filetypes = [("PDF Files", "*.pdf"), ("Text Files", "*.txt")]
        else:
            filetypes = [("Text Files", "*.txt")]

        filename = filedialog.asksaveasfilename(
            title="Save your notes",
            defaultextension=".pdf" if PDF_SUPPORT else ".txt",
            filetypes=filetypes
        )

        if not filename:
            return

        try:
            if filename.endswith(".pdf") and PDF_SUPPORT:
                self.save_as_pdf(text, filename)
            else:
                self.save_as_txt(text, filename)
            messagebox.showinfo("Success", f"Notes saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file: {str(e)}")

    def save_as_pdf(self, text, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(filename)

    def save_as_txt(self, text, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()

# ---------- MAIN ----------
if __name__ == "__main__":
    # Validate Gemini connection
    try:
        test_client = GeminiClient()
        test = test_client.explain("Test connection")
        if "Error" in test:
            raise Exception(test)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Gemini Error", f"Gemini API connection failed: {str(e)}")
        sys.exit(1)
    app = SmartBookApp()
    app.mainloop()
