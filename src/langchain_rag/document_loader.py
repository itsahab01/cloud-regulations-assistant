"""
Document Loader & Semantic Processor for CST Cloud Regulations
Loads 'تنظيمات الحوسبه السحابيه' and produces LangChain Document objects.
"""

import os
import json
from typing import List
from langchain_core.documents import Document

def load_cst_documents(base_dir: str = None) -> List[Document]:
    """
    Loads regulatory documents from 'تنظيمات الحوسبه السحابيه.txt' and structured chunks.
    Returns a list of LangChain Document objects with complete metadata.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    chunks_json_path = os.path.join(base_dir, "data", "cst_chunks.json")
    full_text_path = os.path.join(base_dir, "تنظيمات الحوسبه السحابيه.txt")

    documents: List[Document] = []

    # Priority 1: Load structured regulatory chunks with exact metadata
    if os.path.exists(chunks_json_path):
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            for item in chunks_data:
                content = (
                    f"الوثيقة: تنظيمات تقديم خدمات الحوسبة السحابية (RS10)\n"
                    f"المادة / الجدول: {item.get('section_id')}\n"
                    f"العنوان: {item.get('title')}\n"
                    f"الصفحة: {item.get('page')}\n"
                    f"التصنيف المستهدف: {item.get('classification', 'عام')}\n"
                    f"الجهة: {item.get('entity_type', 'عام')}\n"
                    f"موقع التخزين الإلزامي: {item.get('storage_location', '')}\n"
                    f"فئة مقدم الخدمة: {item.get('provider_class', '')}\n"
                    f"الموافقات: {item.get('approval_required', '')}\n\n"
                    f"نص المادة / البند:\n{item.get('content')}"
                )
                doc = Document(
                    page_content=content,
                    metadata={
                        "id": item.get("id"),
                        "source": "تنظيمات الحوسبه السحابيه",
                        "page": item.get("page"),
                        "section_id": item.get("section_id"),
                        "title": item.get("title"),
                        "classification": item.get("classification"),
                        "storage_location": item.get("storage_location"),
                        "provider_class": item.get("provider_class"),
                        "approval_required": item.get("approval_required"),
                        "legal_articles": item.get("legal_articles", [])
                    }
                )
                documents.append(doc)

    # Also parse pages from the raw text file to ensure 100% comprehensive coverage
    if os.path.exists(full_text_path):
        with open(full_text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Split by page marker [صفحة X]
        pages = raw_text.split("---")
        for p in pages:
            p_strip = p.strip()
            if not p_strip:
                continue
            # Extract page number if present
            page_num = 1
            if "[صفحة " in p_strip:
                try:
                    page_part = p_strip.split("[صفحة ")[1].split("]")[0]
                    page_num = int(page_part.strip())
                except Exception:
                    page_num = 1

            doc = Document(
                page_content=p_strip,
                metadata={
                    "source": "تنظيمات الحوسبه السحابيه",
                    "page": page_num,
                    "title": f"صفحة {page_num} من وثيقة تنظيمات الحوسبة السحابية"
                }
            )
            documents.append(doc)

    return documents
