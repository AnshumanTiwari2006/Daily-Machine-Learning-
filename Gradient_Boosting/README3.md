                 ORIGINAL DATA
                      │
                      ▼
             ┌─────────────────┐
             │ Initial Model   │
             │     f₀(x)       │
             │    mean(y)      │
             └────────┬────────┘
                      │
                      ▼
                Calculate
             Negative Gradient
                      │
                      ▼
             ┌─────────────────┐
             │  Residuals /    │
             │ Pseudo-residuals│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Regression Tree │
             └────────┬────────┘
                      │
                      ▼
              Find leaf regions
                    Rjm
                      │
                      ▼
             Calculate γ for
              every leaf
                      │
                      ▼
             ┌─────────────────┐
             │ Update Model    │
             │                 │
             │ New = Old + γ   │
             └────────┬────────┘
                      │
                      ▼
                New residuals
                      │
                      ▼
                 NEXT TREE
                      │
                      ▼
                    Repeat
                    
                  ORIGINAL DATA
                      │
                      ▼
             ┌─────────────────┐
             │ Initial Model   │
             │     f₀(x)       │
             │    mean(y)      │
             └────────┬────────┘
                      │
                      ▼
                Calculate
             Negative Gradient
                      │
                      ▼
             ┌─────────────────┐
             │  Residuals /    │
             │ Pseudo-residuals│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Regression Tree │
             └────────┬────────┘
                      │
                      ▼
              Find leaf regions
                    Rjm
                      │
                      ▼
             Calculate γ for
              every leaf
                      │
                      ▼
             ┌─────────────────┐
             │ Update Model    │
             │                 │
             │ New = Old + γ   │
             └────────┬────────┘
                      │
                      ▼
                New residuals
                      │
                      ▼
                 NEXT TREE
                      │
                      ▼
                    Repeat