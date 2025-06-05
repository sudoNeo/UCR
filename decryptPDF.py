import os
from PyPDF2 import PdfReader, PdfWriter

PASSWORD = "enterPasswordHere"
INPUT_DIR  = "."           # current folder
OUTPUT_DIR = "decrypted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(".pdf"):
        in_path  = os.path.join(INPUT_DIR, fname)
        out_name = fname[:-4] + "_decrypted.pdf"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        try:
            reader = PdfReader(in_path)
            if reader.is_encrypted:
                reader.decrypt(PASSWORD)
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)

            with open(out_path, "wb") as f_out:
                writer.write(f_out)
            print(f"Decrypted → {out_name}")
        except Exception as e:
            print(f"Failed {fname}: {e}")
