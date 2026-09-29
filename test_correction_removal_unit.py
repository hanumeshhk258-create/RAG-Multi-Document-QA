import sys
import json
from rag.evaluator import correct_and_filter_answer
from rag.vectorstore import load_vectorstore, load_vectorstore_metadata
from rag.embeddings import get_embedding_model
from rag.retriever import retrieve_relevant_chunks, _CROSS_ENCODER

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_hallucination_filtering():
    print("=" * 70)
    print("TESTING HALLUCINATION REMOVAL & CLAIM REWRITING")
    print("=" * 70)

    emb = get_embedding_model("all-MiniLM-L6-v2")
    vs = load_vectorstore(emb, folder_path="vectorstore")

    # Retrieve chunks for CommerceOS
    retrieved_items, _, _, _, _ = retrieve_relevant_chunks(
        vectorstore=vs,
        query="CommerceOS payment processing and architecture",
        top_k=5,
        selected_documents=["CommerceOS_Complete_Project_Report.pdf"]
    )

    # Simulated draft answer containing 1 real fact, 1 partial fact, and 1 hallucinated fact
    draft_answer = (
        "CommerceOS provides secure payment processing with AES-256 encryption.\n"
        "CommerceOS supports multi-region deployment and automatic failover across planetary data centers.\n"
        "CommerceOS natively integrates international cryptocurrency transactions with Bitcoin and Ethereum."
    )

    print("\n--- ORIGINAL DRAFT ANSWER ---")
    print(draft_answer)

    corr_res = correct_and_filter_answer(
        draft_answer=draft_answer,
        retrieved_items=retrieved_items,
        cross_encoder=_CROSS_ENCODER,
        query="CommerceOS payment processing"
    )

    print("\n--- FINAL EVIDENCE-GROUNDED ANSWER ---")
    print(corr_res["final_verified_answer"])

    corr_data = corr_res["answer_correction"]
    print("\n--- ANSWER CORRECTION OBJECT ---")
    print(f"Total Claims: {corr_data['total_claims']}")
    print(f"Supported: {corr_data['supported']}")
    print(f"Partially Supported: {corr_data['partially_supported']}")
    print(f"Unsupported (Removed): {corr_data['unsupported']}")
    print(f"Final Evidence Coverage: {corr_data['final_coverage']}%")
    print(f"Removed Warning: {corr_data['removed_warning']}")

    print("\n--- CLAIMS DETAIL ---")
    for c in corr_data["claims_correction"]:
        print(f"Claim {c['claim_id']}: [{c['status']}] -> Action: {c['action_label']}")
        print(f"  Orig: {c['original_claim']}")
        print(f"  Corr: {c['corrected_claim']}")
        print(f"  Doc: {c['document']} (Page {c['page']})")

    # Assertions
    assert corr_data["unsupported"] >= 1, "Expected at least 1 unsupported claim removed"
    assert corr_data["removed_warning"] is not None, "Expected removal warning banner"
    assert "cryptocurrency" not in corr_res["final_verified_answer"].lower(), "Cryptocurrency statement should be filtered out"
    print("\n✓ ALL ASSERTIONS PASSED!")

if __name__ == "__main__":
    test_hallucination_filtering()
