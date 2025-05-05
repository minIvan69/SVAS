@app.post("/enroll/{user_id}")
async def enroll(user_id: str, file: UploadFile = File(...), tier: Tier = Body(...)):
    # отправить задачу в очередь
    return {"task_id": task.id}

@app.post("/verify")
async def verify(file: UploadFile = File(...), user_id: str = Body(...), tier: Tier = Body(...)):
    # отправить в очередь, ждать или возвращать async task_id
    return {"task_id": task.id}
