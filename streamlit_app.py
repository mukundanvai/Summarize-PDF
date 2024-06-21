import streamlit as st
from PyPDF2 import PdfReader
from app import SummarizationCrew

def main():
    
    # creating streamlit app
    st.header("Agent PDF Summarizer")
    pdf_file = st.file_uploader("Upload PDF File", type="pdf", accept_multiple_files=False)
    submit = st.button("Generate Summary")
    final_text = []
    n=1
    
    if pdf_file and submit:    
        # create a PDF reader object
        pdf_reader = PdfReader(pdf_file)

        # get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # loop through each page and extract the text
        for page_num in range(num_pages):
            # get the page object and extract text from each page
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            final_text.append(text)
            
            # run the agent on each page
            # page_summary = SummarizationCrew.agents_tasks(text=text)
            # summary.append(page_summary)
        
        text = "\n\n".join(final_text)
                    
        final_summary_obj = SummarizationCrew(text=text)
        final_summary = final_summary_obj.run()
        # with open("output.txt", 'w', encoding='utf-8') as f:
        #     f.write(final_summary)
        st.write(final_summary)

if __name__ == "__main__":
    main()