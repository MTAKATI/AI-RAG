import pypdf
import re


class PdfReader:

    def __init__(self, path="./pdfs/AfriConnect.pdf"):
        """
        Initializes the PyPDF reader and placeholder for extracted text.
        """
        self.reader = pypdf.PdfReader(path)
        self.pages_text = ""

    def extract_text(self):
        """
        Extracts, merges, and normalizes text from all PDF pages.
        """
        all_pages_text = []  # store text of each page
        for _, page in enumerate(self.reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_pages_text.append(page_text)
        
        # Join the text from all pages and normalize whitespace
        pdf_text = "\n".join(all_pages_text)
        pdf_text = re.sub(r'\n([ \t]*\n)?[ \t]{2,}', '¶', pdf_text)     # 1. mark paragraph breaks BEFORE space normalization
        pdf_text = re.sub(r' +', ' ', pdf_text)                         # 2. collapse multiple spaces
        pdf_text = re.sub(r'[ \t]*\n[ \t]*', ' ', pdf_text)             # 3. collapse word-wrap newlines → space
        pdf_text = pdf_text.replace('¶', '\n')                          # 4. restore paragraph breaks as \n
        pdf_text = re.sub(r' +', ' ', pdf_text).strip()                 # 5. final space cleanup
        
        print(f"Successfully extracted text. Total characters: {len(pdf_text)}")
        self.pages_text = pdf_text
        return pdf_text

    def extract_small_portion_of_the_pdf(self, start=0, end=None):
        """
        Returns a specific character slice of the extracted text.
        """
        if self.pages_text == "":
            self.extract_text()
        return self.pages_text[start:end]
    
    def get_paragraphs(self):
        """
        Splits the normalized text into clean, individual paragraphs.
        """
        if self.pages_text == "":
            self.extract_text()
        # Corrected: Use .split('\n') instead of calling the string directly
        return [p.strip() for p in self.pages_text.split('\n') if p.strip()]
    

### Testing ground
if __name__ == "__main__":
    pdf_reader = PdfReader()
    pdf_reader.extract_text()
    print("\n--- First 100 characters ---")
    print(pdf_reader.extract_small_portion_of_the_pdf(start=0, end=100))
    print("\n--- First 3 paragraphs ---")
    print(pdf_reader.get_paragraphs()[:3])