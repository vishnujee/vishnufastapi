    # ✅ LOG FINAL DOCUMENTS GOING TO LLM
                # logger.info("📤 FINAL DOCUMENTS BEING SENT TO LLM:")

                # Add this before the llm.invoke call in both RAG and mode-specific sections
                # logger.info("🔍 FINAL PROMPT BEING SENT TO LLM:")
                # for i, (role, content) in enumerate(messages):
                #     logger.info(f"   {i}. {role.upper()}: {content}")
                # for i, doc in enumerate(processed_docs, 1):
                #     logger.info(f"📄 Document {i}/{len(processed_docs)}:")
                #     logger.info(f"   Source: {doc.metadata.get('source', 'unknown')}")
                #     logger.info(f"   Content Type: {doc.metadata.get('content_type', 'unknown')}")
                #     logger.info(f"   Content Preview: {doc.page_content}...")
                #     logger.info(f"   Full Content Length: {len(doc.page_content)} chars")
                #     logger.info("   ---")

                # ✅ PREPARE LLM CHAIN WITH CHAT HISTORY
                # Build messages with conversation history