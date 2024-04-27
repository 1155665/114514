try:
    req = HTTPSConnection(host)
    req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
    resp = req.getresponse()
    print(resp.read())
except Exception as err:
    print(err)