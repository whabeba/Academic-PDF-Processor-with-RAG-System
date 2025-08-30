import streamlit as st
import fitz
import io
import json
import os
import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from datetime import datetime
import textwrap
import zipfile
import tempfile
import shutil
import time
import numpy as np
import sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
API_KEY = "AIzaSyDcVBoTfLAvVOqS7V9FSHxwIU-cKV6UsFo"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Initialize sentence transformer for embeddings
@st.cache_resource
def load_embedding_model():
    return sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2',device="cpu")

class Summary(BaseModel):
    paper_name: str = Field(..., min_length=10, max_length=300,
                            description="The actual name of the text.")
    authors_name: Dict[str, str] = Field(..., min_items=1,
                                        description="The names of the authors are shared on text")
    summary_in_english: List[str] = Field(..., min_items=30, max_items=100,
                                         description="Summarized text in english in key points about the text (50-100 sentences).")
    summary_in_egyptian: List[str] = Field(..., min_items=30, max_items=100,
                                          description="Summarized text in egyptian arabic in key points about the text (50-100 sentences")
    equations: str = Field(..., min_length=10, max_length=300,
                          description="The equations in the text.")

class Translation(BaseModel):
    translated_text: str = Field(..., description="The full text translated to Egyptian Arabic")

class RAGSystem:
    def __init__(self, summary_data, original_text):
        self.embedding_model = load_embedding_model()
        self.summary_data = summary_data
        self.original_text = original_text
        self.embeddings = None
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for the summary points"""
        if not self.summary_data or not self.summary_data.get('summary_in_english'):
            return
        
        summary_points = self.summary_data['summary_in_english']
        self.embeddings = self.embedding_model.encode(summary_points)
    
    def retrieve_relevant_points(self, query, top_k=3):
        """Retrieve the most relevant summary points for a given query"""
        if self.embeddings is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top_k most relevant points
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_points = [(i, self.summary_data['summary_in_english'][i], similarities[i]) 
                          for i in top_indices]
        
        return relevant_points
    
    def generate_response(self, query):
        """Generate a response to the user's query based on the summary"""
        relevant_points = self.retrieve_relevant_points(query)
        
        if not relevant_points:
            return "I couldn't find relevant information in the summary to answer your question."
        
        # Prepare context from relevant points
        context = "\n".join([f"Point {idx+1}: {point}" for idx, point, _ in relevant_points])
        
        # Generate prompt for the model
        prompt = f"""
        You are an expert assistant explaining academic papers. Based on the following summary points from the paper "{self.summary_data.get('paper_name', 'Unknown Paper')}", answer the user's question.
        
        Summary Points:
        {context}
        
        User Question: {query}
        
        Provide a detailed explanation based on these points. If the points don't contain enough information to answer the question, please mention that.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while trying to answer your question."

class PDFProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "output"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    def extract_text_and_images(self) -> tuple:
        doc = fitz.open(self.pdf_path)
        full_text = ""
        image_paths = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = f"{self.output_dir}/images/page{page_num}_img{img_index}.{image_ext}"
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                image_paths.append(image_path)
        return full_text, image_paths
    
    def generate_summary(self, text: str) -> Optional[Summary]:
        summarization_prompt = [
            {"role": "user", "parts": [{"text": "\n".join([
                "You are an expert in AI and text localization to Egyptian Arabic (colloquial). Perform the following steps:",
                "1. Read the provided English text.",
                "2. Extract the paper name and authors if present.",
                "3. Summarize the text in English in 50-100 key points (each point being a sentence).",
                "4. Translate the summary into Egyptian Arabic (colloquial) while keeping all numbers, equations, names of people, places, companies, and technical/scientific terms in English.",
                "5. Extract any equations present in the text.",
                "6. Follow the provided schema to generate a JSON.",
                "",
                "## Text:",
                text.strip(),
                "",
                "## Pydantic Schema:",
                json.dumps(Summary.model_json_schema(), ensure_ascii=False),
                "",
                "## Output:",
                "```json"
            ])}]}
        ]
        try:
            response = model.generate_content(summarization_prompt)
            response_text = response.text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            summary_data = json.loads(json_str)
            return Summary(**summary_data)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def translate_to_egyptian(self, text: str) -> Optional[str]:
        translation_prompt = [
            {"role": "user", "parts": [{"text": "\n".join([
                "You are an expert in translation and text adaptation to Egyptian Arabic (colloquial). Your task is:",
                "1. Take the academic or formal text (in English) and convert it into natural, fluent Egyptian Arabic.",
                "2. If the text contains names of people, places, companies, scientific terms, or technical terms, keep them in English (do not translate them).",
                "3. Preserve the original meaning without removing any important information, but if there are overly formal or complex sentences, simplify them so they are easy to understand in Egyptian Arabic.",
                "4. Do not change any numbers or data; keep them as they are.",
                "5. Do not summarize the text, just translate it.",
                "6. The output will be fluent Egyptian Arabic.",
                "",
                "## Text:",
                text.strip(),
                "",
                "## Pydantic Schema:",
                json.dumps(Translation.model_json_schema(), ensure_ascii=False),
                "",
                "## Output:",
                "```json"
            ])}]}
        ]
        try:
            response = model.generate_content(translation_prompt)
            response_text = response.text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            translation_data = json.loads(json_str)
            return translation_data["translated_text"]
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return None
    
    def save_summary_to_pdf(self, summary: Summary) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"{self.output_dir}/summary_{timestamp}.pdf"
        doc = fitz.open()
        page = doc.new_page()
        width, height = page.rect.width, page.rect.height
        margin = 50
        y_position = margin
        title = "Paper Summary Report"
        title_width = fitz.get_text_length(title, fontsize=20, fontname="helv")
        page.insert_text(
            (width / 2 - title_width / 2, y_position),
            title,
            fontsize=20,
            fontname="helv",
            color=(0.2, 0.2, 0.8)
        )
        y_position += 50
        paper_name = summary.paper_name
        paper_name_width = fitz.get_text_length(paper_name, fontsize=16, fontname="helv")
        page.draw_rect([margin - 10, y_position - 5, margin + paper_name_width + 10, y_position + 25],
                      color=(0.8, 0.8, 0.9), fill=(0.9, 0.9, 0.95), width=1)
        page.insert_text(
            (margin, y_position),
            f"Paper: {paper_name}",
            fontsize=16,
            fontname="helv",
            color=(0.1, 0.1, 0.1)
        )
        y_position += 40
        authors_text = "Authors:"
        page.insert_text(
            (margin, y_position),
            authors_text,
            fontsize=14,
            fontname="helv",
            color=(0.1, 0.1, 0.1)
        )
        y_position += 25
        for author, affiliation in summary.authors_name.items():
            author_text = f"• {author} ({affiliation})"
            page.insert_text(
                (margin + 10, y_position),
                author_text,
                fontsize=12,
                fontname="helv",
                color=(0.0, 0.0, 0.0)
            )
            y_position += 20
        y_position += 20
        page.insert_text(
            (margin, y_position),
            "English Summary",
            fontsize=16,
            fontname="helv",
            color=(0.1, 0.1, 0.1)
        )
        y_position += 5
        page.draw_line((margin, y_position), (width - margin, y_position), color=(0.5, 0.5, 0.5), width=0.5)
        y_position += 20
        for i, point in enumerate(summary.summary_in_english, 1):
            if y_position > height - margin - 50:
                page = doc.new_page()
                y_position = margin
            page.insert_text(
                (margin, y_position),
                f"{i}. ",
                fontsize=12,
                fontname="helv",
                color=(0.2, 0.2, 0.8)
            )
            wrapped_text = textwrap.fill(point, width=90)
            text_lines = wrapped_text.split('\n')
            for line in text_lines:
                if y_position > height - margin - 20:
                    page = doc.new_page()
                    y_position = margin
                page.insert_text(
                    (margin + 25, y_position),
                    line,
                    fontsize=12,
                    fontname="helv",
                    color=(0.0, 0.0, 0.0)
                )
                y_position += 18
            y_position += 5
        page = doc.new_page()
        y_position = margin
        page.insert_text(
            (margin, y_position),
            "Equations",
            fontsize=16,
            fontname="helv",
            color=(0.1, 0.1, 0.1)
        )
        y_position += 5
        page.draw_line((margin, y_position), (width - margin, y_position), color=(0.5, 0.5, 0.5), width=0.5)
        y_position += 20
        equations = summary.equations
        equation_lines = equations.split('\n')
        max_line_width = 0
        for line in equation_lines:
            line_width = fitz.get_text_length(line, fontsize=14, fontname="helv")
            max_line_width = max(max_line_width, line_width)
        box_width = max_line_width + 40
        box_height = len(equation_lines) * 25 + 20
        page.draw_rect([margin - 10, y_position - 10, margin + box_width, y_position + box_height],
                      color=(0.8, 0.8, 0.9), fill=(0.95, 0.95, 0.98), width=1)
        for line in equation_lines:
            page.insert_text(
                (margin, y_position),
                line,
                fontsize=14,
                fontname="helv",
                color=(0.0, 0.0, 0.0)
            )
            y_position += 25
        timestamp_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        timestamp_width = fitz.get_text_length(timestamp_text, fontsize=10, fontname="helv")
        page.insert_text(
            (width - timestamp_width - margin, height - 30),
            timestamp_text,
            fontsize=10,
            fontname="helv",
            color=(0.5, 0.5, 0.5)
        )
        doc.save(pdf_path)
        doc.close()
        logger.info(f"Summary saved to PDF: {pdf_path}")
        return pdf_path
    
    def process_pdf(self) -> Dict:
        logger.info(f"Processing PDF: {self.pdf_path}")
        full_text, image_paths = self.extract_text_and_images()
        summary = self.generate_summary(full_text)
        pdf_path = None
        if summary:
            pdf_path = self.save_summary_to_pdf(summary)
        egyptian_translation = self.translate_to_egyptian(full_text)
        results = {
            "summary": summary.dict() if summary else None,
            "egyptian_translation": egyptian_translation,
            "extracted_images": image_paths,
            "original_text": full_text,
            "summary_pdf_path": pdf_path
        }
        with open(f"{self.output_dir}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Processing complete. Results saved to {self.output_dir}")
        return results

def create_images_zip(image_paths):
    if not image_paths:
        return None
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for img_path in image_paths:
            if os.path.exists(img_path):
                zip_file.write(img_path, os.path.basename(img_path))
    
    zip_buffer.seek(0)
    return zip_buffer

def display_chat_interface():
    """Display the chat interface for the RAG system"""
    st.subheader("Ask Questions About the Paper")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask a question about the paper summary"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        if 'rag_system' in st.session_state:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Get response from RAG system
                response = st.session_state.rag_system.generate_response(prompt)
                
                # Simulate stream of response with chunks
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("Please process a PDF first before asking questions.")

def main():
    st.set_page_config(
        page_title="PDF Academic Paper Processor with RAG",
        page_icon="📚",
        layout="wide"
    )
    st.title("Academic PDF Processor & Translator with RAG System")
    st.markdown("Upload your academic PDF, get structured summaries, and ask questions about the content")
    
    # Initialize session state variables
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    temp_dir = tempfile.mkdtemp()
                    pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    processor = PDFProcessor(pdf_path, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    results = processor.process_pdf()
                    
                    st.session_state.processed_results = results
                    st.session_state.processor = processor
                    
                    # Initialize RAG system if summary is available
                    if results and results.get("summary"):
                        st.session_state.rag_system = RAGSystem(
                            results["summary"], 
                            results.get("original_text", "")
                        )
                    
                    st.success("PDF processed successfully!")
        
        st.subheader("Download Options")
        
        if st.session_state.processed_results:
            results = st.session_state.processed_results
            
            if results["summary_pdf_path"] and os.path.exists(results["summary_pdf_path"]):
                with open(results["summary_pdf_path"], "rb") as file:
                    st.download_button(
                        label="Download Summary PDF",
                        data=file.read(),
                        file_name="summary.pdf",
                        mime="application/pdf"
                    )
            
            if results["extracted_images"]:
                zip_buffer = create_images_zip(results["extracted_images"])
                if zip_buffer:
                    st.download_button(
                        label="Download Images (ZIP)",
                        data=zip_buffer,
                        file_name="extracted_images.zip",
                        mime="application/zip"
                    )
            
            json_data = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download Full Results (JSON)",
                data=json_data,
                file_name="results.json",
                mime="application/json"
            )
    
    with col2:
        if st.session_state.processed_results:
            results = st.session_state.processed_results
            summary = results["summary"]
            
            if summary:
                st.subheader("Paper Information")
                st.write(f"**Paper Name:** {summary['paper_name']}")
                
                authors_text = ", ".join([f"{k} ({v})" for k, v in summary['authors_name'].items()])
                st.write(f"**Authors:** {authors_text}")
                
                st.write(f"**Images Extracted:** {len(results['extracted_images'])}")
                st.write(f"**Equations:** {summary['equations']}")
                
                tab1, tab2, tab3, tab4 = st.tabs(["English Summary", "Egyptian Summary", "Translation", "Ask Questions"])
                
                with tab1:
                    st.subheader("English Summary")
                    for i, point in enumerate(summary['summary_in_english'], 1):
                        st.write(f"{i}. {point}")
                
                with tab2:
                    st.subheader("ملخص باللهجة المصرية")
                    for i, point in enumerate(summary['summary_in_egyptian'], 1):
                        st.write(f"{i}. {point}")
                
                with tab3:
                    st.subheader("Full Text Translation")
                    if results['egyptian_translation']:
                        st.write(results['egyptian_translation'])
                    else:
                        st.write("Translation not available")
                
                with tab4:
                    display_chat_interface()
            else:
                st.error("Failed to process the PDF. Please try again.")
        else:
            st.info("Upload and process a PDF to see results here.")

if __name__ == "__main__":
    main()