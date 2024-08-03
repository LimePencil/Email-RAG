document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chatbot-form");
  const chatHistory = document.getElementById("chat-history");
  const questionInput = document.getElementById("question");
  let chatHistoryData = [];

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value;
    if (!question) return;

    // 사용자의 질문을 채팅 기록에 추가
    addMessageToChat("user", question);
    questionInput.value = "";

    // 로딩 스피너 표시
    document.querySelector(".loading-spinner").style.display = "block";

    try {
      // 서버에 질문을 전송하고 응답을 기다림
      const response = await fetch("/chat/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ history: chatHistoryData, question: question }),
      });
      const data = await response.json();

      // 서버의 응답을 채팅 기록에 추가
      addMessageToChat("bot", data.answer);
      chatHistoryData.push([question, data.answer]);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      // 로딩 스피너 숨김
      document.querySelector(".loading-spinner").style.display = "none";
    }
  });

  function addMessageToChat(sender, message) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", sender);
    messageElement.innerText = message;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
});
