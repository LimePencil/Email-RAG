document
  .getElementById("chatbot-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const question = document.getElementById("question").value;
    // 로딩 스피너 표시
    document.querySelector(".loading-spinner").style.display = "block";

    try {
      const response = await fetch("/chat/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: question }),
      });
      const data = await response.json();

      // 로딩 스피너 숨기기
      document.querySelector(".loading-spinner").style.display = "none";

      //console.log(data);
      const answerDiv = document.getElementById("answer");
      answerDiv.style.display = "block";
      answerDiv.innerText = `Answer: ${data.answer}`;
    } catch (error) {
      console.error("There was an error!", error);

      // 로딩 스피너 숨기기
      document.querySelector(".loading-spinner").style.display = "none";
    }
  });
document.querySelector(".loading-spinner").style.display = "none";
