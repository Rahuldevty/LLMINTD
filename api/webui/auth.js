const authTitle = document.getElementById("authTitle");
const authMessage = document.getElementById("authMessage");
const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");
const showLoginButton = document.getElementById("showLoginButton");
const showSignupButton = document.getElementById("showSignupButton");
const googleButtonWrap = document.getElementById("googleButtonWrap");
const googleHint = document.getElementById("googleHint");

function setAuthMode(mode) {
  const loginActive = mode === "login";
  loginForm.hidden = !loginActive;
  signupForm.hidden = loginActive;
  showLoginButton.classList.toggle("active-tab", loginActive);
  showSignupButton.classList.toggle("active-tab", !loginActive);
  authTitle.textContent = loginActive ? "Welcome Back" : "Create Your Account";
  authMessage.textContent = loginActive
    ? "Login, then choose testing or deployment."
    : "Create an account first. After signup, choose testing or deployment.";
}

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

function goToChoice() {
  window.location.href = "/ui/choice";
}

async function loadSession() {
  const data = await requestJson("/auth/session", { method: "GET" });
  if (data.user) {
    goToChoice();
  }
}

async function loadGoogleConfig() {
  try {
    const data = await requestJson("/auth/google/config", { method: "GET" });
    if (!data.enabled || !data.client_id) {
      googleHint.textContent = "Google login is disabled until a Google Client ID is configured.";
      return;
    }
    googleHint.textContent = "Use Google Sign-In for faster onboarding.";
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => {
      google.accounts.id.initialize({
        client_id: data.client_id,
        callback: async (response) => {
          try {
            await requestJson("/auth/google", {
              method: "POST",
              body: JSON.stringify({ credential: response.credential }),
            });
            goToChoice();
          } catch (error) {
            googleHint.textContent = error.message;
          }
        },
      });
      google.accounts.id.renderButton(googleButtonWrap, {
        theme: "outline",
        size: "large",
        shape: "pill",
        width: 280,
        text: "continue_with",
      });
    };
    document.head.appendChild(script);
  } catch (error) {
    googleHint.textContent = error.message;
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(loginForm);
  try {
    await requestJson("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email: formData.get("email"),
        password: formData.get("password"),
      }),
    });
    goToChoice();
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(signupForm);
  try {
    await requestJson("/auth/signup", {
      method: "POST",
      body: JSON.stringify({
        full_name: formData.get("full_name"),
        email: formData.get("email"),
        password: formData.get("password"),
      }),
    });
    goToChoice();
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

showLoginButton.addEventListener("click", () => setAuthMode("login"));
showSignupButton.addEventListener("click", () => setAuthMode("signup"));

const requestedMode = new URLSearchParams(window.location.search).get("mode");
setAuthMode(requestedMode === "login" ? "login" : "signup");
loadSession().catch(() => {});
loadGoogleConfig();
