from __future__ import annotations

import logging

from voice_agent.kb import KnowledgeBase
from voice_agent.settings import load_settings
from voice_agent.utils.doc_ingestion import ingest_pdf_into_kb


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    kb = KnowledgeBase.from_settings(settings)
    ingestion = ingest_pdf_into_kb(kb, pdf_url=settings.kb_source_pdf_url)
    print(
        "seeded_documents={seeded_documents} source_url={source_url} collection={collection} total={total}".format(
            seeded_documents=ingestion.document_count,
            source_url=ingestion.source_url,
            collection=settings.kb_collection_name,
            total=kb.count(),
        )
    )
