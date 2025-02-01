import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
import base64

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
            "serial_number": "",
            "deleted": False,
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

        # Sort contours by position (top to bottom, left to right)
        def sort_contours(cnts):
            # Get bounding boxes
            boxes = [cv2.boundingRect(c) for c in cnts]
            
            # Calculate average height
            avg_height = sum(h for x, y, w, h in boxes) / len(boxes)
            
            # Group boxes into rows with more precise y-coordinate grouping
            row_tolerance = avg_height * 0.3  # 30% of average height as tolerance
            rows = {}
            
            for i, (x, y, w, h) in enumerate(boxes):
                # Find the row this box belongs to
                assigned = False
                for row_y in rows.keys():
                    if abs(y - row_y) < row_tolerance:
                        rows[row_y].append((i, x, y))
                        assigned = True
                        break
                
                if not assigned:
                    rows[y] = [(i, x, y)]
            
            # Sort rows by y-coordinate and boxes within rows by x-coordinate
            sorted_indices = []
            for row_y in sorted(rows.keys()):
                # Sort boxes in this row by x-coordinate
                row_boxes = sorted(rows[row_y], key=lambda b: b[1])
                sorted_indices.extend([i for i, x, y in row_boxes])
            
            return [cnts[i] for i in sorted_indices]

        # Sort contours by position
        contours = sort_contours(contours)
        
        voters = []
        for idx, contour in enumerate(contours, 1):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 5000:  # Adjust threshold as needed
                continue
                
            print(f"Processing contour {idx} (area: {area})")
            x, y, w, h = cv2.boundingRect(contour)
            cell = gray[y:y+h, x:x+w]
            
            # Enhanced DELETED stamp detection
            is_deleted = False
            
            # Create a copy of cell for stamp detection
            stamp_region = cell.copy()
            
            # Additional delete text variations that might appear when text is broken/partial
            delete_variations = [
                "DELETE", "DELETED", "DEL E", "DELET", "LET E", "DE TED", 
                "DEL ED", "D TED", "DELE", "ETED", "D E L", "DE L",
                "DELT", "DEL T", "DELE D"
            ]
            
            # Try multiple preprocessing approaches
            preprocessing_steps = [
                # Original grayscale
                lambda img: img,
                # Enhanced contrast
                lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0),
                # Red channel isolation (for red stamps)
                lambda img: cv2.split(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[2],
                # Adaptive thresholding
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
            ]
            
            # Multiple threshold values
            thresholds = [130, 150, 170, 190]
            
            for preprocess in preprocessing_steps:
                processed = preprocess(stamp_region)
                
                for threshold in thresholds:
                    # Apply different thresholding methods
                    _, binary = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)
                    _, inv_binary = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY_INV)
                    
                    # Try both normal and inverted images
                    for img in [binary, inv_binary]:
                        # Apply additional preprocessing
                        # 1. Denoise
                        denoised = cv2.fastNlMeansDenoising(img)
                        
                        # 2. Dilate to connect broken text
                        kernel = np.ones((2,2), np.uint8)
                        dilated = cv2.dilate(denoised, kernel, iterations=1)
                        
                        # Try OCR with different PSM modes
                        for psm in [6, 7, 8, 3]:
                            text = pytesseract.image_to_string(
                                dilated,
                                lang='eng',
                                config=f'--psm {psm} --oem 3'
                            ).strip().upper()
                            
                            # Check for delete variations
                            if any(var in text for var in delete_variations):
                                is_deleted = True
                                print(f"Found DELETE stamp in box {idx} (threshold={threshold}, psm={psm})")
                                print(f"Detected text: {text}")
                                break
                        
                        if is_deleted:
                            break
                    
                    if is_deleted:
                        break
                
                if is_deleted:
                    break
            
            # Extract serial number (top left corner)
            serial_region = cell[0:h//4, 0:w//4]
            serial_number = pytesseract.image_to_string(
                serial_region,
                lang='eng',
                config='--psm 6 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            # Extract voter ID (at bottom right of the box)
            voter_id_region = cell[3*h//4:h, 2*w//3:w]
            
            # Enhance voter ID image
            voter_id_region = cv2.resize(voter_id_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            voter_id_binary = cv2.adaptiveThreshold(
                voter_id_region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
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
            
            # Add serial number and deleted status
            parsed_details["serial_number"] = serial_number
            parsed_details["deleted"] = is_deleted

            voter_info = {
                "page_number": page_num,
                "voter_id": voter_id,
                "details": parsed_details
            }
            voters.append(voter_info)

        return voters

    def is_metadata_page(self, page_np):
        """Check if page contains metadata text markers"""
        metadata_markers = [
            "परिवर्धन पूरक सूची",
            "फोटोयुत िनवाचक",
            "नज़री नशा",
            "नगरपािलका िनवाचन",
            "िवलोपन","पूरक सूची",
            "विलोपित", "संशोधित",
            "संक्षिप्त विवरण"
        ]
        
        # Convert to grayscale if needed
        if len(page_np.shape) == 3:
            gray = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = page_np
            
        # Extract text using Tesseract
        text = pytesseract.image_to_string(
            gray, 
            lang='script/Devanagari+eng',
            config='--psm 6'
        ).strip()
        
        # Check if any metadata marker is present
        return any(marker in text for marker in metadata_markers)

    def check_vilopit(self, page_np):
        """Check if page contains विलोपित text"""
        if len(page_np.shape) == 3:
            gray = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = page_np
            
        text = pytesseract.image_to_string(
            gray, 
            lang='script/Devanagari+eng',
            config='--psm 6'
        ).strip()
        
        return "विलोपित" in text

    def process_pdf(self):
        """Process the entire PDF"""
        print(f"\nStarting PDF processing from page {self.resume_page}")
        try:
            self.data["metadata"] = {"pages": []}  # Reset metadata structure

            # Get total number of pages
            print("Counting total pages...")
            pages = convert_from_path(self.pdf_path, dpi=300)
            total_pages = len(pages)
            print(f"Total pages in PDF: {total_pages}")

            # Process pages until विलोपित is found
            for page_num in range(self.resume_page, total_pages + 1):
                print(f"\nProcessing page {page_num}/{total_pages}")
                
                page = convert_from_path(
                    self.pdf_path,
                    first_page=page_num,
                    last_page=page_num,
                    dpi=300
                )[0]
                page_np = np.array(page)
                
                # Check for विलोपित before processing
                if self.check_vilopit(page_np):
                    print(f"Found विलोपित on page {page_num}. Stopping further processing.")
                    return True  # Return True as we processed some pages
                
                # Check if it's a metadata page
                if self.is_metadata_page(page_np):
                    print(f"Found metadata on page {page_num}")
                    text = pytesseract.image_to_string(
                        page_np, 
                        lang='script/Devanagari+eng',
                        config='--psm 6'
                    ).strip()
                    self.data["metadata"]["pages"].append({
                        "page_number": page_num,
                        "content": text
                    })
                    continue

                voters = self.extract_voter_info(page_num, page_np)
                print(f"Extracted {len(voters)} voter records from page {page_num}")
                self.data["voters"].extend(voters)
                self.save_progress()
                self.resume_page = page_num + 1
                
            print("\nPDF processing completed successfully!")
            return True
            
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

def process_pdf_folder(pdf_folder, output_folder="output"):
    """Process all PDF files in the folder and its subfolders"""
    print(f"\nProcessing PDF files in folder: {pdf_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                
                # Create output path maintaining folder structure
                rel_path = os.path.relpath(root, pdf_folder)
                output_dir = os.path.join(output_folder, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                
                output_json = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
                
                print(f"\nProcessing PDF: {pdf_path}")
                print(f"Output will be saved to: {output_json}")
                
                try:
                    extractor = VoterListExtractor(pdf_path, output_json=output_json)
                    if not extractor.process_pdf():  # If विलोपित found, skip to next PDF
                        print(f"Skipping {pdf_path} due to विलोपित")
                        continue
                    print(f"Successfully processed {pdf_path}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    continue

def main():
    pdf_folder = "pdf"
    output_folder = "output"  # Specify output folder
    print(f"\nStarting voter list extraction from folder: {pdf_folder}")
    print(f"Output files will be saved to: {output_folder}")
    
    try:
        process_pdf_folder(pdf_folder, output_folder)
        print("All PDF processing completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Check individual JSON files in output folder for results")

if __name__ == "__main__":
    main()
