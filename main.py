import gradio as gr
import fitz  # PyMuPDF
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# Global Variables
# ==============================
pdf_text_storage = ""  # Stores extracted PDF text
print("🚀 Initializing StudyMate...")

# ==============================
# Load IBM Granite Model
# ==============================
print("🔄 Loading IBM Granite model... This may take a while (large model download).")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-2b-instruct",
    device_map="auto",  # Auto device placement
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print("✅ IBM Granite model loaded.")


# ==============================
# Functions
# ==============================
def process_pdf_file(file_obj):
    """Extracts text from uploaded PDF and stores it globally."""
    global pdf_text_storage
    try:
        file_path = Path(str(file_obj))
        if not file_path.exists():
            return "❌ Error: File does not exist."

        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

        pdf_text_storage = text  # Save for Q&A
        return f"✅ Extracted {len(text)} characters from PDF."
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"


def ask_question(user_question):
    """Answer user question using Granite model + stored PDF text."""
    global pdf_text_storage
    if not pdf_text_storage:
        return "❌ No document processed yet. Upload and process a PDF first."

    try:
        # Create prompt: PDF context + question
        messages = [
            {"role": "system", "content": "You are StudyMate, an AI assistant for students. Use the given PDF content to answer."},
            {"role": "user", "content": f"PDF Content:\n{pdf_text_storage[:4000]}\n\nQuestion: {user_question}"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=200)
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return answer.strip()

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def generate_study_plan(weeks, hours_per_week, topic_hint):
    """Generate a custom study plan using Granite model."""
    try:
        hint_text = f" focusing on {topic_hint}" if topic_hint else ""
        messages = [
            {"role": "system", "content": "You are StudyMate, an academic assistant."},
            {"role": "user", "content": f"Create a {weeks}-week study plan with {hours_per_week} hours per week{hint_text}."}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=300)
        plan = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return plan.strip()

    except Exception as e:
        return f"❌ Error generating study plan: {str(e)}"


# ==============================
# Gradio UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("## 📘 StudyMate\nUpload a PDF, ask questions, and generate custom study plans.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", type="filepath")
            process_btn = gr.Button("Process PDF")
            process_output = gr.Textbox(label="Processed Output", interactive=False)

        with gr.Column():
            gr.Markdown("### Ask a Question")
            question_input = gr.Textbox(label="Your Question")
            ask_btn = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📅 Generate Study Plan")
            weeks_input = gr.Number(label="Weeks", value=4, precision=0)
            hours_input = gr.Number(label="Hours per Week", value=5, precision=0)
            topic_input = gr.Textbox(label="Topic hint (optional)")
            study_btn = gr.Button("Generate Study Plan")
            study_output = gr.Textbox(label="Study Plan", interactive=False)

    # Wiring buttons
    process_btn.click(fn=process_pdf_file, inputs=pdf_input, outputs=process_output)
    ask_btn.click(fn=ask_question, inputs=question_input, outputs=answer_output)
    study_btn.click(fn=generate_study_plan, inputs=[weeks_input, hours_input, topic_input], outputs=study_output)

# ==============================
# Launch
# ==============================
if __name__ == "__main__":
    demo.launch()
