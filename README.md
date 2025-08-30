# Academic PDF Processor & Translator with RAG System

A powerful web application that processes academic PDF papers, generates structured summaries in both English and Egyptian Arabic, and features a Retrieval-Augmented Generation (RAG) system for interactive question-answering about the paper content.

## Features

### PDF Processing
- **Text Extraction**: Extract text content from PDF files
- **Image Extraction**: Extract and save images from PDF files
- **Structured Summarization**: Generate comprehensive summaries with key points
- **Bilingual Support**: Create summaries in both English and Egyptian Arabic
- **Equation Extraction**: Identify and extract mathematical equations from the text

### Translation Capabilities
- **Full Text Translation**: Translate the entire paper content to Egyptian Arabic
- **Preserved Terminology**: Keep technical terms, names, and equations in their original form
- **Colloquial Adaptation**: Convert formal academic text to natural, fluent Egyptian Arabic

### RAG (Retrieval-Augmented Generation) System
- **Semantic Search**: Ask questions about the paper content
- **Context-Aware Answers**: Get responses based on the actual paper content
- **Interactive Chat**: Chat with the system to explore the paper in depth
- **Relevant Point Retrieval**: The system identifies the most relevant summary points for each question

### User Interface
- **Modern Web Interface**: Built with Streamlit for an intuitive user experience
- **Tabbed Navigation**: Organized display of different content types
- **Download Options**: Download summaries as PDF, images as ZIP, or complete results as JSON
- **Responsive Design**: Works well on different screen sizes

## Screenshots

*(Add screenshots of your application here)*

## Demo

Check out a live demo of the application: [Link to your deployed app]

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

Step 2: Install Dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Set Up API Key
GOOGLE_API_KEY=your_gemini_api_key_here
Usage
streamlit run app.py