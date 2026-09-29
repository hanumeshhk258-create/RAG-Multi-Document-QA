"""
Script to create sample PDFs with structured tables for testing Table Extraction RAG feature.
"""
import os

def generate_table_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    os.makedirs("documents", exist_ok=True)
    pdf_path = "documents/ML_Algorithms_Comparison.pdf"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    elements = []
    
    # Title & Intro
    elements.append(Paragraph("Machine Learning Algorithms and Taxonomy", title_style))
    elements.append(Paragraph(
        "Machine learning models can be broadly categorized according to their learning paradigm, "
        "training data requirements, and primary strengths in statistical data analysis. "
        "The following structured table provides a comprehensive summary of widely used algorithms.",
        body_style
    ))
    elements.append(Spacer(1, 10))
    
    # Table 1: Algorithms Comparison Table
    table_data = [
        ["Algorithm", "Type", "Advantage", "Primary Use Case"],
        ["KNN", "Supervised", "Simple", "Classification and Regression"],
        ["K-Means", "Unsupervised", "Clustering", "Customer Segmentation"],
        ["Decision Tree", "Supervised", "Explainability", "Rule-based Decisions"],
        ["PCA", "Unsupervised", "Dimensionality Reduction", "Feature Compression"],
        ["Random Forest", "Supervised", "High Accuracy", "Ensemble Prediction"],
        ["DBSCAN", "Unsupervised", "Arbitrary Shapes", "Spatial Density Clustering"]
    ]
    
    t = Table(table_data, colWidths=[100, 95, 130, 185])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f1f5f9')]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph(
        "KNN is a simple supervised algorithm relying on distance metrics to classify nearest instances. "
        "K-Means is an unsupervised clustering algorithm that iteratively partitions data points into K clusters. "
        "Decision Trees provide superior explainability for supervised tabular tasks.",
        body_style
    ))
    
    doc.build(elements)
    print(f"Successfully generated {pdf_path}")

if __name__ == "__main__":
    generate_table_pdf()
