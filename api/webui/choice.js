const logoutButton = document.getElementById("logoutButton");
const userName = document.getElementById("userName");
const userEmail = document.getElementById("userEmail");
const userAvatar = document.getElementById("userAvatar");
const openTesterButton = document.getElementById("openTesterButton");
const openDeployButton = document.getElementById("openDeployButton");

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function setUser(user) {
  if (!user) {
    window.location.href = "/ui/login?mode=login";
    return;
  }
  userName.textContent = user.full_name;
  userEmail.textContent = user.email;
  userAvatar.textContent = (user.full_name || user.email || "AI").slice(0, 2).toUpperCase();
}

logoutButton.addEventListener("click", async () => {
  await requestJson("/auth/logout", { method: "POST", body: JSON.stringify({}) });
  window.location.href = "/ui";
});

openTesterButton.addEventListener("click", () => {
  window.location.href = "/ui/test";
});

openDeployButton.addEventListener("click", () => {
  window.location.href = "/ui/deploy";
});

requestJson("/auth/session", { method: "GET" })
  .then((data) => setUser(data.user))
  .catch(() => setUser(null));
