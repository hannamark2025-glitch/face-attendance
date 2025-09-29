from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24
    API_PREFIX: str = "/api"

    # Postgres connection string will come from Render
    DB_URL: str

    SENDGRID_API_KEY: str
    FROM_EMAIL: str

    class Config:
        env_file = ".env"

settings = Settings()
