from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import UserSignUp, Credentials, get_db
from src.queries.orm import Orm

router = APIRouter(tags=["auth"])


@router.post("/signup")
async def signup(
    user_data: UserSignUp, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.register_user(
            session,
            firstName=user_data.firstName,
            lastName=user_data.lastName,
            login=user_data.login,
            email=user_data.email,
            password=user_data.password,
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                },
                "error": result,
            }
        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
            },
            "error": None,
        }

    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
            },
            "error": f"Ошибка при регистрации пользователя: {str(e)}",
        }


@router.post("/signin")
async def signin_user(
    credentials: Credentials, session: AsyncSession = Depends(get_db)
):
    try:
        result = await Orm.signin_user(
            session,
            login=credentials.login,
            password=credentials.password,
        )

        if isinstance(result, str):
            return {
                "userData": {
                    "id": None,
                    "firstName": None,
                    "lastName": None,
                    "email": None,
                },
                "error": result,
            }

        return {
            "userData": {
                "id": result["id"],
                "firstName": result["firstName"],
                "lastName": result["lastName"],
                "email": result["email"],
            },
            "error": None,
        }

    except Exception as e:
        return {
            "userData": {
                "id": None,
                "firstName": None,
                "lastName": None,
                "email": None,
            },
            "error": f"Ошибка при входе: {str(e)}",
        }
