import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path

# Set Tesseract path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

class VoterListExtractor:
    def __init__(self, pdf_path, output_json="voter_data.json", resume_page=1):
        print(f"Initializing extractor for PDF: {pdf_path}")
        self.pdf_path = pdf_path
        self.output_json = output_json
        self.resume_page = resume_page
        self.data = self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing JSON data if available"""
        if os.path.exists(self.output_json):
            print(f"Loading existing data from {self.output_json}")
            with open(self.output_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        print("No existing data found, starting fresh")
        return {"metadata": {}, "voters": []}

    def save_progress(self):
        """Save current progress to JSON file"""
        print(f"Saving progress to {self.output_json}")
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def parse_voter_details(self, text):
        """Parse voter details text into structured data"""
        print("Parsing voter details")
        details = {
            "name": "",
            "father_name": "",
            "husband_name": "",
            "house_number": "",
            "age": "",
            "gender": "",
            "raw_text": text
        }
        
        # Split text into lines and clean
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"Raw text lines: {lines}")  # Debug print
        
        # Try to extract structured information
        for line in lines:
            if "नाम" in line or "नम" in line:
                details["name"] = line.split(":")[-1].strip() if ":" in line else line
            elif "पिता :" in line:
                details["father_name"] = line.split(":")[-1].strip()
            elif "पति :" in line:
                details["husband_name"] = line.split(":")[-1].strip()
            elif "मकान" in line or "घर" in line:
                details["house_number"] = line.split(":")[-1].strip() if ":" in line else line
            # Handle combined gender and age line
            elif any(gender_term in line for gender_term in ["लिग", "िलंग"," िलंग", "लिंग"]):
                print(f"Found gender/age line: {line}")  # Debug print
                
                # Split line by "आयु" to separate gender and age
                parts = line.split("आयु")
                
                # Extract gender from first part
                gender_part = parts[0]
                if ":" in gender_part:
                    gender_text = gender_part.split(":")[-1].strip()
                    if "पुरुष" in gender_text or "पुरष" in gender_text or "पुरुस" in gender_text:
                        details["gender"] = "पुरुष"
                    elif "महिला" in gender_text or "स्त्री" in gender_text or "महीला" in gender_text:
                        details["gender"] = "महिला"
                
                # Extract age from second part if it exists
                if len(parts) > 1:
                    age_part = parts[1]
                    if ":" in age_part:
                        age = ''.join(filter(str.isdigit, age_part.split(":")[-1]))
                        details["age"] = age if age else ""
                
                print(f"Detected gender: {details['gender']}, age: {details['age']}")  # Debug print
        
        return details

    def process_voter_id(self, voter_id):
        """Process voter ID with special handling for first 3 letters"""
        # Clean the voter ID first
        voter_id = ''.join(c for c in voter_id if c.isalnum())
        
        if len(voter_id) >= 3:
            # First 3 characters should be letters
            prefix = voter_id[:3].upper()
            numbers = voter_id[3:]
            
            # Replace Z with 2 only in the number portion
            numbers = numbers.replace('Z', '2').replace('z', '2')
            
            return prefix + numbers
        return voter_id

    def extract_voter_info(self, page_num, image):
        """Extract voter information from a single page"""
        print(f"\nProcessing page {page_num}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Threshold the image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours on page {page_num}")

        voters = []
        for idx, contour in enumerate(contours, 1):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 5000:  # Adjust threshold as needed
                continue
                
            print(f"Processing contour {idx} (area: {area})")
            x, y, w, h = cv2.boundingRect(contour)
            cell = gray[y:y+h, x:x+w]

            # Extract voter ID (at bottom right of the box)
            voter_id_region = cell[3*h//4:h, 2*w//3:w]
            
            # Enhance voter ID image
            # Apply multiple preprocessing steps to improve OCR
            # 1. Increase resolution
            voter_id_region = cv2.resize(voter_id_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 2. Apply adaptive thresholding
            voter_id_binary = cv2.adaptiveThreshold(
                voter_id_region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # 3. Denoise
            voter_id_binary = cv2.fastNlMeansDenoising(voter_id_binary)
            
            # Extract text with specific configurations for voter ID
            voter_id = pytesseract.image_to_string(
                voter_id_binary,
                lang='eng',
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()
            
            # Process voter ID with special handling for first 3 letters
            voter_id = self.process_voter_id(voter_id)
            print(f"Voter ID: {voter_id}")

            # Extract voter details (main area, excluding photo region)
            details_region = cell[0:h, 0:2*w//3]
            voter_details_text = pytesseract.image_to_string(
                details_region, 
                lang='script/Devanagari+eng',
                config='--psm 6'
            ).strip()
            
            # Parse details into structured format
            parsed_details = self.parse_voter_details(voter_details_text)

            voter_info = {
                "page_number": page_num,
                "voter_id": voter_id,
                "details": parsed_details
            }
            voters.append(voter_info)

        return voters

    def process_pdf(self):
        """Process the entire PDF"""
        print(f"\nStarting PDF processing from page {self.resume_page}")
        try:
            if not self.data["metadata"]:
                self.extract_metadata()

            # Get total number of pages
            print("Counting total pages...")
            pages = convert_from_path(self.pdf_path, dpi=300)
            total_pages = len(pages)
            print(f"Total pages in PDF: {total_pages}")

            # Process each page starting from resume_page
            for page_num in range(self.resume_page, total_pages + 1):
                print(f"\nProcessing page {page_num}/{total_pages}")
                
                # Convert page to image
                page = convert_from_path(
                    self.pdf_path,
                    first_page=page_num,
                    last_page=page_num,
                    dpi=300
                )[0]
                
                # Skip first two metadata pages
                if page_num <= 2:
                    print(f"Skipping metadata page {page_num}")
                    continue

                # Extract voter information
                voters = self.extract_voter_info(page_num, np.array(page))
                print(f"Extracted {len(voters)} voter records from page {page_num}")
                self.data["voters"].extend(voters)

                # Save progress after each page
                self.save_progress()
                self.resume_page = page_num + 1
                
            print("\nPDF processing completed successfully!")
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print(f"Progress saved. Resume from page {self.resume_page}")
            raise

    def extract_metadata(self):
        """Extract metadata from first two pages"""
        print("Extracting metadata...")
        
        # Convert first two pages
        pages = convert_from_path(
            self.pdf_path,
            first_page=1,
            last_page=2,
            dpi=300
        )

        metadata = {}
        for i, page in enumerate(pages, 1):
            # Convert to numpy array
            page_np = np.array(page)
            
            # Convert to grayscale if needed
            if len(page_np.shape) == 3:
                page_np = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)
            
            # Extract text using Tesseract with Devanagari script
            text = pytesseract.image_to_string(
                page_np, 
                lang='script/Devanagari+eng',
                config='--psm 6'
            )
            metadata[f"page_{i}"] = text.strip()
            print(f"Extracted metadata from page {i}")

        self.data["metadata"] = metadata
        self.save_progress()

def main():
    # Initialize extractor
    pdf_path = "pdf/Antim.pdf"
    print(f"\nStarting voter list extraction from: {pdf_path}")
    
    try:
        extractor = VoterListExtractor(pdf_path)
        extractor.process_pdf()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Check the voter_data.json file for partial results")

if __name__ == "__main__":
    main()
