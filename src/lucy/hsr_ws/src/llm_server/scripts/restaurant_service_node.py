#!/usr/bin/env python3

import rospy
import ollama
from llm_server.srv import ExtractOrder, ExtractOrderResponse


class OllamaOrderExtractor:
    def __init__(self, model_name: str = "granite3.1-moe:1b"):
        """
        Initialize the Ollama Order Extractor
        """
        self.model_name = model_name

    def extract_order_items(self, input_text: str) -> str:
        """
        Extract dish names from the order text
        """
        try:
            system_prompt = """
                You are an AI assistant that extracts only the names of dishes from customer orders.
                Given an input containing an order, return only the dish names in a comma-separated format.
                Do not include extra words, quantities, or descriptions.
                Your output should contain only the dish names.
                """

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                options={"temperature": 0.1}
            )

            return response["message"]["content"].strip()

        except Exception as e:
            rospy.logerr(f"Ollama extraction error: {e}")
            return ""


class OrderExtractionService:
    def __init__(self):
        rospy.loginfo("Starting Order Extraction Service...")

        model_name = rospy.get_param("~ollama_model", "granite3.1-moe:1b")

        self.extractor = OllamaOrderExtractor(model_name)

        self.service = rospy.Service(
            "extractOrder",
            ExtractOrder,
            self.handle_request
        )

        rospy.loginfo("Order Extraction Service Ready")

    def handle_request(self, req):
        """
        Handle incoming service requests
        """
        rospy.loginfo(f"Received order text: {req.text}")

        items = self.extractor.extract_order_items(req.text)

        rospy.loginfo(f"Extracted items: {items}")

        return ExtractOrderResponse(items)


def main():
    rospy.init_node("order_extraction_service")

    OrderExtractionService()

    rospy.spin()


if __name__ == "__main__":
    main()


# works with counts:
# #!/usr/bin/env python3

# import rospy
# import ollama
# import json
# from llm_server.srv import ExtractOrder, ExtractOrderResponse


# class OllamaOrderExtractor:

#     def __init__(self, model_name="granite3.1-moe:1b"):
#         self.model_name = model_name

#     def extract_order_items(self, input_text: str):

#         system_prompt = """
# You extract food orders from text.

# Return ONLY valid JSON.

# Schema:
# {
#  "items": [
#    {"name": "dish_name", "count": number}
#  ]
# }

# Rules:
# - Count quantities if mentioned
# - If quantity not mentioned assume count = 1
# - Merge duplicate items
# - Only include food or drink items
# - Output JSON only
# """

#         try:

#             response = ollama.chat(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": input_text}
#                 ],
#                 options={"temperature": 0}
#             )

#             content = response["message"]["content"].strip()

#             # Remove markdown if model adds it
#             if content.startswith("```"):
#                 content = content.replace("```json", "").replace("```", "").strip()

#             data = json.loads(content)

#             return data

#         except Exception as e:
#             rospy.logwarn(f"LLM extraction failed: {e}")

#             return {"items": []}


# class OrderExtractionService:

#     def __init__(self):

#         rospy.loginfo("Starting Order Extraction Service")

#         model = rospy.get_param("~ollama_model", "granite3.1-moe:1b")

#         self.extractor = OllamaOrderExtractor(model)

#         self.service = rospy.Service(
#             "extract_order_items",
#             ExtractOrder,
#             self.handle_request
#         )

#         rospy.loginfo("Order Extraction Service Ready")

#     def handle_request(self, req):

#         text = req.text.strip()

#         rospy.loginfo(f"Received order text: {text}")

#         order_data = self.extractor.extract_order_items(text)

#         order_json = json.dumps(order_data)

#         rospy.loginfo(f"Extracted order: {order_json}")

#         return ExtractOrderResponse(order_json)


# def main():

#     rospy.init_node("order_extraction_service")

#     OrderExtractionService()

#     rospy.spin()


# if __name__ == "__main__":
#     main()
