from typing import Dict, Any

class PromptTemplates:
    """Holds the prompt templates for the agent. """

    @staticmethod
    def intent_detection_prompt(user_input: str) -> str:
        """System prompt that detects intention of question of user and provides possible actions using the agent's tools."""
        return f"""Determine the user's intent based on their {user_input} and suggest relevant actions using the available tools.

                    Posible intents: 
                    - check if automation is already done by Yarado, document_search 
                    - suggest an automation method that Yarado can provide, document_search, Google_search

                    Explicitation of the user query to gather more context if needed -> that's a condition to stay in the loop to determine intent.
                    
                    
                    Only give the intent a name as result.
                    """

    @staticmethod
    def document_search_prompt(cases_summary: list) -> str:
        """System prompt that summarizes the relevant documents found for the user query."""
        if not cases_summary:
            return "No relevant documents found."

        return f"""DYou have the following similar cases found in the internal knowledge base:

                    {cases_summary}

                    Please summarize the key relevant points from these cases that can help answer the user's query.
                    If the cases are not sufficient, indicate that more information is needed.

                    Also look for similar cases in the knowledge base.
                    - Document the findings and any relevant details that can help the user determine in RPA of AI automation is possible within their context.
                    - Identify any gaps in the information provided by the cases and suggest areas for further exploration.


                    Provide a concise yet informative summary
                    """

    @staticmethod
    def web_search_prompt(results_summary: list) -> str:
        """System prompt that summarizes the relevant web search results found for the user query."""
        if not results_summary:
            return "No relevant web search results found."

        return f"""You have the following relevant web search results:

                    {results_summary}

                    Please summarize the key relevant points from these results that can help answer the user's query.
                    If the results are not sufficient, indicate that more information is needed.

                    Also look for similar cases in the knowledge base.
                    - Document the findings and any relevant details that can help the user determine in RPA of AI automation is possible within their context.
                    - Identify any gaps in the information provided by the results and suggest areas for further exploration.

                    Provide a concise yet informative summary
                    """
    

    @staticmethod
    def compliance_check_prompt(compliance_text: str) -> str:
        """System prompt that summarizes the compliance requirements based on the provided text."""
        return f"""Based on the following information about compliance requirements:

            {compliance_text}

            Please summarize the key relevant points from this information and provide concrete advice.
            Indicate if there are any ambiguities or additional checks needed.

            Provide the answer in clear, understandable language.

            Also include any relevant legal references or guidelines that may apply.
            - [GDPR](https://gdpr-info.eu/)
            - [ISO 27001](https://www.iso.org/iso-27001-information-security.html)
            - [NEN 7510](https://www.nen.nl/nen-7510-2020-nl-2020-12-01.htm)
            """
    
    @staticmethod
    def synthesize_results_prompt(combined_context: str, user_question: str) -> str:
        """System prompt that summarizes the synthesized results based on the provided text."""
        return f"""Based on the following synthesized results:

            {combined_context}

            Please summarize the key relevant points from this information and provide concrete advice.
            Indicate if there are any ambiguities or additional checks needed.

            Check if the synthesized results adequately address the user question:
            {user_question}
            If not, please specify what additional information is needed.
            Provide any relevant details that can help address the gaps.
             """
