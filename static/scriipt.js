document.getElementById("send-button").addEventListener("click", async () => {
  const userInput = document.getElementById("user-input");
  const query = userInput.value;
  if (!query) return;

  // Display user query in chat box
  addMessageToChat("You: " + query);
  userInput.value = "";

  // Send query to Flask backend
  const response = await fetch("/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: query }),
  });

  const data = await response.json();
  addMessageToChat("Bot: " + data.response);
});

// Function to add messages to chat box
function addMessageToChat(message) {
  const chatBox = document.getElementById("chat-box");
  chatBox.innerHTML += "<div>" + message + "</div>";
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
}
