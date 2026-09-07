def run_model_comparison(client_gemini, collection, embed_model, topics, evaluation_queries, candidate_models):
    all_results = []
    total = len(candidate_models) * len(evaluation_queries)
    count = 0

    for model_name in candidate_models:
        for item in evaluation_queries:
            count += 1
            try:
                result = answer_query(
                    client_gemini=client_gemini,
                    query=item["query"],
                    collection=collection,
                    embed_model=embed_model,
                    topics=topics,
                    gen_model=model_name,
                )
                result["expected_id"] = item["expected_id"]
                result["retrieval_correct"] = result["retrieved_id"] == item["expected_id"]
                all_results.append(result)

                # imprime o bloco assim que esse resultado fica pronto
                status = "✓" if result["retrieval_correct"] else "✗"
                print(f"\n[{count}/{total}] {status} Modell: {model_name}")
                print(f"Frage: {result['query']}")
                print(f"Thema erkannt: {result['topic']}")
                print(f"Intro: {result['intro']}")
                print("Phrasen:")
                for phrase in result["phrases"]:
                    print(f"  • {phrase}")
                print("-" * 80)

            except Exception as e:
                print(f"\n[{count}/{total}] ✗ [DEBUG] Erro com {model_name}: {type(e).__name__}: {e}")

    return all_results