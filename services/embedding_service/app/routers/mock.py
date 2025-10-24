from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()

class MockRequest(BaseModel):
    message: str

class MockResponse(BaseModel):
    response: str

@router.post("/mock", response_model=MockResponse)
def repeat_message(request: MockRequest):
    mock_response = f"ROBOT: You said\n {request.message}"
    return MockResponse(response=mock_response)