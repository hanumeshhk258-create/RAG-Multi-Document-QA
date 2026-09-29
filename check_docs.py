from rag.pdf_loader import load_pdf_file
docs, _ = load_pdf_file('documents/CommerceOS_Complete_Project_Report.pdf', 'CommerceOS_Complete_Project_Report.pdf')
for d in docs:
    p = d.metadata.get('page')
    if p in [3, 4, 5]:
        print(f"=== PAGE {p} ===")
        print(d.page_content)
        print("-" * 50)
