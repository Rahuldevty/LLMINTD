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

requestJson("/auth/session", { method: "GET" })
  .then((data) => {
    if (data.user) {
      window.location.href = "/ui/choice";
    }
  })
  .catch(() => {});
