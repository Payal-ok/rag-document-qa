# Introduction to REST APIs

REST stands for Representational State Transfer. A REST API lets programs talk to a server over HTTP. The server exposes resources, such as users or tasks, and each resource has a URL.

## HTTP methods

Clients use HTTP methods to say what they want to do. GET reads a resource and does not change anything. POST creates a new resource. PUT replaces an existing resource, while PATCH changes only some of its fields. DELETE removes a resource.

## Status codes

The server answers every request with a status code. The code 200 means the request succeeded. The code 201 means a new resource was created. The code 400 means the client sent a bad request, for example invalid JSON. The code 401 means the client is not authenticated, and 404 means the resource does not exist. Codes in the 500 range, such as 500, mean the server itself failed.

## Good design habits

Use nouns for URLs, such as `/tasks/42`, and let the HTTP method describe the action. Return JSON, and validate every input before using it. A well designed API is stateless, which means each request carries all the information the server needs, so servers can be added or removed easily. Documenting every endpoint with example requests and responses saves other developers a lot of time.
