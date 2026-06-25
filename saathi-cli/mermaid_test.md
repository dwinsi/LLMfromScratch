# Mermaid Test

## Test 1 — minimal graph

```mermaid
graph LR
    A[hello] --> B[world]
```

## Test 2 — subgraph

```mermaid
graph LR
    subgraph SUB [my subgraph]
        A[hello]
    end
    A --> B[world]
```

## Test 3 — edge labels

```mermaid
graph LR
    A[hello] -->|some label| B[world]
```

## Test 4 — subgraph with edge labels

```mermaid
graph LR
    subgraph SUB [my subgraph]
        A[hello]
    end
    A -->|some label| B[world]
```

## Test 5 — multiple subgraphs with cross edges

```mermaid
graph LR
    subgraph ONE [first]
        A[hello]
    end
    subgraph TWO [second]
        B[world]
    end
    A --> B
```

## Test 6 — minimal sequenceDiagram

```mermaid
sequenceDiagram
    participant A
    participant B
    A->>B: hello
    B-->>A: world
```

## Test 7 — sequenceDiagram with loop

```mermaid
sequenceDiagram
    participant A
    participant B
    loop my loop
        A->>B: hello
        B-->>A: world
    end
```

## Test 8 — sequenceDiagram with alias

```mermaid
sequenceDiagram
    participant A as my alias
    participant B
    A->>B: hello
```
