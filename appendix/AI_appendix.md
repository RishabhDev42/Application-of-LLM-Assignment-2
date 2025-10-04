# AI Appendix

Queries to Gemini 2.5 Pro:
1. Here are details of my assignment, add these to your context for the thread:
2. i have written the code for naive rag, help me come up with an experimentation excel
3. note: SQuAD Metrics:

Exact Match (EM): 1.6340

F1 Score: 22.7411 are the metrics that i am using along with ragas metrics metrics=[

        faithfulness,

        answer_relevancy,

        context_recall,

        context_precision,

    ] 
4. query id, question and ground truth are useless because I am using a standardised dataset
5. Generate helpful comments for my code
Note: I used Gemini's autocompletion while coding along with debugging assistance while developing on Google Colab
    
Queries to GPT-4.1:
1. Break down my architecture into points for clarity, keep the descriptions of each step short and concise
2. Generate a puml diagram for the architecture that I have described in my README
3. Help me rewrite the experimental analysis for this project using the experiment csv file. It should include statistical support.
4. What are some scalability, deployment recommendations, limitations that I should include in my readme
5. Help me come up with an executive summary for my readme