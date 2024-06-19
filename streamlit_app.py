import streamlit as st
from PyPDF2 import PdfReader
from app import SummarizationCrew

def main():
    st.header("Agent PDF Summarizer")
    pdf_file = st.file_uploader("Upload PDF File", type="pdf", accept_multiple_files=False)
    submit = st.button("Generate Summary")

    if pdf_file and submit:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        summary_crew = SummarizationCrew(text=text)
        summary = summary_crew.run()

        st.write("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()