import streamlit as st
from PyPDF2 import PdfReader
from app import SummarizationCrew

def main():
    st.header("Agent PDF Summarizer")
    pdf_file = st.file_uploader("Upload PDF File", type="pdf", accept_multiple_files=False)
    submit = st.button("Generate Summary")
    summary = []
    if pdf_file and submit:    
        # Create a PDF reader object
        pdf_reader = PdfReader(pdf_file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # Loop through each page and extract the text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            
            # Extract the text from the page
            text = page.extract_text()
            
            # Print the text
            page_summary = SummarizationCrew.agents_tasks(text=text)
            summary.append(page_summary)
        text = "\n\n".join(summary)
        final_summary_obj = SummarizationCrew(text=text)
        final_summary = final_summary_obj.run()
        st.write(final_summary)

        # Close the PDF file
        pdf_file.close()

if __name__ == "__main__":
    main()