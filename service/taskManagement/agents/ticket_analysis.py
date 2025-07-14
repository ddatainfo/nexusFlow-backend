from typing import Dict
from sentence_transformers import SentenceTransformer
import chromadb
from utils.state import conversation_states  # ✅ FIX added here

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.PersistentClient(path=r"/home/ddata/work/nantha/chatbot_api/pdf_service/jira_vector_db")
collection = client.get_collection(name="both_ticket_embeddings")
SIMILARITY_THRESHOLD = 0.75
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

class TicketAnalysisAgent:
    def __init__(self, convo_id: str):
        self.convo_id = convo_id
        self.required_fields = ["title", "description", "priority"]
        self.state = conversation_states.get(convo_id, {}).get("fields", {})

    def analyze_ticket(self) -> str:
        result = self._validate_ticket_info(self.state)
        if result["is_complete"]:
            print("\n🧠 Ticket Analysis Result:")
            print("Ticket Analyzed ✅")
            return self.perform_similarity_search()
        else:
            print("\n Ticket Validation Failed:")
            print("Missing Fields:", ", ".join(result["missing_fields"]))
            for suggestion in result["suggestions"]:
                print("-", suggestion)
            return "Ticket Incomplete"

    def _validate_ticket_info(self, ticket_info: Dict) -> Dict:
        suggestions = []
        missing_fields = []

        for field in self.required_fields:
            value = ticket_info.get(field, "")
            if not value or str(value).strip() == "":
                missing_fields.append(field)

        title = ticket_info.get("title", "").strip()
        if title and len(title) < 5:
            suggestions.append("Title Suggestion: Title is too short or unclear. Consider providing more specific details.")

        description = ticket_info.get("description", "").strip()
        if description and len(description) < 15:
            suggestions.append("Description Suggestion: Description may be too brief. Add more detail to clarify the issue.")

        return {
            "is_complete": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "suggestions": suggestions
        }

    def perform_similarity_search(self):
        query = f"{self.state['title']} {self.state['description']}"
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )

        similarity_details = []
        matches_found = False
        convo_state = conversation_states[self.convo_id]

        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            if distance <= DISTANCE_THRESHOLD:
                matches_found = True
                metadata = results["metadatas"][0][i]
                doc = results["documents"][0][i]

                ticket_type = metadata.get("type", "N/A").capitalize()
                ticket_id = metadata.get("id", "N/A")
                ticket_key = metadata.get("key", "N/A")
                summary = metadata.get("summary") or metadata.get("name") or doc.split("\n")[0]
                similarity_score = round(1.0 - distance, 4)

                print(f"🎫 {ticket_type} - Similarity: {similarity_score}")
                print(f"  ID   : {ticket_id}")
                print(f"  Key  : {ticket_key}")
                print(f"  Title: {summary}")
                print("-" * 50)

                similarity_details.append(
                    f"🎫 {ticket_type} - Similarity: {similarity_score}\n"
                    f"  ID   : {ticket_id}\n"
                    f"  Key  : {ticket_key}\n"
                    f"  Title: {summary}\n"
                    + "-" * 50
                )

        if matches_found:
            full_message = "\n".join([
                "🔎 Top Matching Tickets (similarity ≥ 0.75):",
                *similarity_details,
                "",
                "I've found similar tickets already reported. Would you still like to create a new ticket?",
                "b$YES$b",
                "b$NO$b"
            ])
        else:
            full_message = "\n".join([
                "❌ No similar tickets found.",
                "Would you like to create a new ticket for this issue?",
                "b$YES$b",
                "b$NO$b"
            ])

        convo_state["conversation"].append({"role": "assistant", "content": full_message})
        print(f"Starlistant: {full_message}")
        convo_state["awaiting_ticket_confirmation"] = True
        return full_message