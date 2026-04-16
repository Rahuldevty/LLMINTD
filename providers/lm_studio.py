from providers.openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        *,
        logger,
        base_url: str,
        api_key: str,
        model_name: str | None,
        request_timeout: int,
        auto_discover_model: bool = True,
    ) -> None:
        super().__init__(
            name="lm_studio",
            logger=logger,
            base_url=base_url,
            api_key=api_key or "lm-studio",
            model_name=model_name,
            request_timeout=request_timeout,
            auto_discover_model=auto_discover_model,
        )
