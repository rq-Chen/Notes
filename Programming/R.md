# R Programming

## Data & Operation

### Data structures

- All data structures are essentially objects with certain attributes

  - e.g., arrays and matrices are simply vectors with `dim` and `dimnames` attributes
  - The attributes can be viewed by `attributes(x)` or set by `attributes(x) <- myNamedList`
    - However, that will print all the values and sometimes may be super long
    - You should use `str(x)` for a shortened version instead
  - A single attribute can be set by `attr(x, name, value)` or `attr(x, name) <- value`, as well as in the constructor function (e.g. `arr <- array(data, dim = xxx, dimnames = xxxx)`)
  - In may cases, the attribute can also be viewed or set using accessor functions, e.g. `names(x) <- xxx`
  - `class` is a special attribute
    - can be checked by `class()`
    - `numeric`, `list`, `matrix`, `array`, `data.frame`, `function`, `formula`, etc.
    - You can check the available methods for a class by `methods(class = "myclass")`
  - The datatype of the elements in the structure can be checked by `typeof()`
    - `double`, `numeric` (integer), `logical`, `character`, `raw` (bit), `complex`, `language` (e.g., for formulas) and so on
    - The function `as.xxx()` provides many conversions.
    - `NA` means missing data and `NaN` is something like 0/0, and they can be tested by `is.na()` (for both) and `is.nan()` (for `NaN` only)

- Data structures:

  - Vectors: 1D sequences of the same datatype
  - Arrays and matrices: similar to matrix in MATLAB
  - Lists: similar to list + dict in Python
  - Factors: similar to categories in MATLAB
  - Data frames: similar to table
  - Functions
  - Formulas

- Vector:

  - Data can be grouped with function `c()`, e.g. `x <- c(1, 2, 3, 4, 5)` renders a numeric vector
  - You can also combine vectors in this way.
  - To get the union of multiple vectors, use `union()`
  - `x <- c(1, '2')` renders a character vector `["1", "2"]`
  - `seq` and `:` expression also create vectors.

- Array:
  - Constructed by array constructor: `array(datavec, dim = dimvec, dimnames = dimnamesvec)`.
  - Get dimensions by `dim()` function

- List:
  - Get length by `length()` function

- Matrix: `m = matrix(c(1,2,3,4), ncol = 2, nrow = 2, byrow = T)`

- Dataframe:

  - `data.frame(col1 = dat1, col2 = dat2, ...)`
  - Note: there seems to be a bug when using `mydfs <- lapply(mylist, as.data.frame)`, consider using `mydfs <- lapply(mylist, function(x) list(as.data.frame(x)))` and then `mydfs <- lapply(mydfs, function(x) x[[1]])` instead

- Formula:

  - Formula is a kind of unevaluated expression to characterize the relationship between variables

  - Formula is usually used in modeling (e.g., `lm`, `glm`, `lme4` and `brms`) and plotting (`ggplot2`)

  - Formula is of type "language" with two or three elements in the following order (1 - 2/3):

    - The tilde operator `~`
    - The left hand side (if any): e.g. `y`
    - The right hand side: e.g. `x + b`

    For example, `~ x + b` is a one-sided formula and `y ~ a` is a two-sided formula

  - Formula has two attributes:

    - `class`: `"formula"`
    - `.Environment`: the R global environment when the formula was created

  - Right hand side: construct the model

    - `+` for adding terms (not arithmatic summation)
    - `-` for removing terms
    - `:` for interaction terms
    - `*` for crossing: `x1 * x2` is the same as `x1 + x2 + x1 : x2`
    - `1` for the intercept (by default added, can be removed with `- 1` or `+ 0`)
    - `^` for restricting the degree of interactions: `(x1 + x2 + x3) ^ 2` includes all variables and two-way interactions
    - `%in%` for nested effects: `x1 + x2 %in% x1` is the same as `x1 + x1 : x2`
    - `/` is a shortcut: `x1 / x2` is the same as `x1 + x1 : x2`
    - `I()` for arithmatic operations: `I(x1 ^ 2)` includes the square of `x1`

  - Usually, the terms will be the names of the columns in a dataframe (or other types of namespace), and we input the formula and the dataframe to the fitting function (e.g., `lm`, `lmer`); otherwise the fitting function will look for the terms in `.Environment` of the formula


### Index

- `1:30` is the same as in MATLAB; `seq(1, 100, by = 3)` is equivalent to `1:3:100` in MATLAB.

- `length()` might not behave like you thought, e.g. `length()` of a 2\*1 dataframe is 1 rather than 2

- There are three ways for indexing:

  - Single bracket
  - Double bracket:
    - Rarely used for vectors and matrices. The main difference with `[]` is that it will drop the names of the dimensions.
    - For lists, `[[]]` gives you a single element, while `[]` gives you a list containing the element(s)
  - `var$name` for named list or dataframe

- Index is column-based and started from 1, similar to MATLAB.

- Index is similar to MATLAB but use a square bracket instead. Besides, you can use negative index to exclude items, e.g. `x[-(1:5), 6]`. You can also use logical index.

- Unlike in MATLAB, you can use matrix as index in R, where each row contains the *n* indices for one element.

- You can assign names to a variable and accessed them with attribute values:

  ```R
  fruit <- c(5, 10, grape = 1, 20)
  names(fruit) <- c("orange", "banana", "apple", "peach")
  lunch <- fruit[c("apple","orange")]
  
  # or by dimnames
  mat <- array(1:4, dim = c(2,2))
  dimnames(mat) <- list(c('x', 'y'), c('x', 'y'))
  subMat <- mat['x','y']  # subMat = 3
  ```

### Operation

- Assigment is like `x <- y` or `y -> x` (assigning y to x)

- `%%` and `%/%` are mod and div respectively

- `&` and `|` are not short-circuit while `&&` and `||` are

- `&` and `|` are VECTORIZED while `&&` and `||` are NOT!

- `*`, `%*%`, `%o%` and `%x%` are elementwise product, matrix multiplication, outer product and kronecker product respectively

- You can check whether an item is in a list by `x %in% y`

- You can clip or extend (by `NA`) a vector by directly modifying its length: `length(y) <- 3`

- You can reshape a vector (e.g. into a matrix) by directly modifying its attribute `dim` similarly.

- You can add up two vector with different length:

  - Which means repeating the short vector for several (maybe not integral) times before element-wise operation.
  - As an example, `labs <- paste0(c("X","Y"), 1:10)` will render a vector of `c("X1", "Y2", "X3", "Y4", "X5", "Y6", "X7", "Y8", "X9", "Y10")`.
  - **IMPORTANT: this is how vector inputs are processed in all base R functions (see below)!**
  - Example:
    - If `X` is a 10\*2 matrix/dataframe and `v` is a 1\*10 (or 10\*1) vector, then `X / v` will devide X's each row by the corresponding element in v
    - However, if `X` is the same while `v` is a 1\*2 vector, `X / v` will NOT divide X's each column by the corresponding element in V; instead, it divides the (2n-1)-th elements (column-based flattening) by `v[[1]]` and (2n)-th elements by `v[[2]]`
    - If you want to acheive something like column-based division, you need to use `sweep(X, 2, v, FUN = "/")` (1 for row and 2 for column)

- Matrices and arrays can be combined with `cbind()` for columns or `rbind()` for rows.

- Matrix multiplication: `%*%`; element-wise multiplication: `*`

- Transpose: `t()`

- String concatenation:

  - `paste(..., sep = " ", collapse = NULL)` converts all inputs to strings and concatenates them

    - `sep` seperates the elements of different inputs

    - If specified, `collapse` concatenates the outputs into a single string

    - Example:

      ```R
      paste(c('X','Y'), 1:5, sep = '_')
      # "X_1" "Y_2" "X_3" "Y_4" "X_5"
      
      paste(c('X','Y'), 1:5, sep = '_', collapse = ' and ')
      # “X_1 and Y_2 and X_3 and Y_4 and X_5”
      ```

  - `paste0` use `sep = ""` by default


## Control Flow & Function

```R
if (something) {
  blablabla
} else if (something) {
  blablabla
} else {
  blablabla
}

for (a in b) {
  something
}

while (somthing) {
  blablabla
  break
  next
}

function_name <- function(arg_1, arg_2) {
  Function body
  return(something)  # Or simply `something`
}
```

- How function with **vector input** works in R:

  ```R
  func(c(1, 2), c(1, 2, 3), matrix(c(1, 2, 3, 4), nCol = 2)) ==
    [[func(1, 1, 1), func(2, 2, 2)],
     [func(1, 3, 3), func(2, 1, 4)]]
  ```

  In other words, the output size is equal to the largest input size (same shape if it's multidimensional), and the shorter inputs are looped again and again (column-based).

- You can apply a function too every element of a list and get a list output of the same length by `sapply()`.

- `ifelse(test, true, false)` will always return a value of the same shape of `test`:

  - So if your `test` returns a simple logical value, only the first element of `true` and `false` will be returned!
  - In other words, `ifelse()` is NOT a ?-expression


## ggplot2

- `ggplot2` is part of `tidyverse` along with `dplyr`, `tidyr`, `readr`, `tibble` and `purrr`

- Important feature of `tidyverse` functions:

  - Each column is a variable in the namespace of the tibble (dataframe), therefore:
    - If you need to refer to a column with non-standard names (e.g., with space), you need to quote it in backticks `` `column name` ``
    - `tidyverse` functions use column names WITHOUT quotations (either as parameter, a single argument or an element in an array)!
  - If you want to use any external variables in a `tidyverse` function:
    - Generally speaking, you can use `get("myvar", .env)` or `.env$myvar`
      - part of `rlang` library, automatically imported with `dplyr` (but not `ggplot2`, you need to import it manually)
      - search for the variable in the environment when the `tidyverse` function is called
    - However, if this variable is a char vector of a COLUMN NAME:
      - You MUST use `.data[[myvar]]`, otherwise `ggplot2::aes()` won't work
  - See the [Programming with dplyr](https://dplyr.tidyverse.org/articles/programming.html) vignette and the [Using ggplot2 in packages](https://ggplot2.tidyverse.org/articles/ggplot2-in-packages.html) vignette, also:
    - https://stackoverflow.com/a/69488093
    - and https://stackoverflow.com/a/55524126
    - and https://tidyselect.r-lib.org/reference/faq-external-vector.html
  
- The fundamental idea of ggplot2 is very similar to `d3.js`, that you should map data elements to geometric objects.

  In d3 this is done by the concept of selection:

  ```javascript
  d3.selectAll('tag1').data(myData).append('tag2')
      .attr('attr1', mapFunc)
      .attr('attr2', C)
  	.attr(...);
  ```

  While in ggplot2 it's done by aesthethic mapping.

  ```R
  myData = tibble(attr1 = c(...), attr2 = c(...), ...)
  ggplot(data = myData, mapping = aes(x = attr1, y = attr2, glbAttr1 = attr3, ...))
  	+ geom_elem1(locAttr1 = C, aes(locAttr2 = attr4, ...))
  	+ geom_elem2(...)
  	+ ...
  	+ geom_elemn(...)
  ```

- You need a scaling fuction for mapping. Just like the `d3.scaleXXX().domain().range()` or `d3.interpolateXXX()`, in ggplot2 we have `scale_ELEM_TYPE()`.

- A very useful tool to arrange suplots is `patchwork`, see https://ggplot2-book.org/arranging-plots.html

## Other Packages

- `dplyr` provides extra grammar to manipulate dataframes:
  - Verbs (operations):
    - `filter()` (for rows), `select()` (for columns), `slice()`, `mutate()` and so on
    - Grouping and group-by-group operations (essentially enabling high-dimensional vectorized operations in 2D dataframes)
    - The syntax for all functions are similar, see [the doc](https://dplyr.tidyverse.org/articles/dplyr.html#patterns-of-operations)
  - The pipe:
    - `%>%` operator can chain expressions
    - e.g., `var1 %>% fun1(param2, param3) %>% fun2(param2a)`  is the same as `fun2(fun1(var1, param2, param3), param2a)` but more readable
  - It's not easy to do operations over rows in `dplyr`, but relatively easy over columns:
    - Therefore, you can first `pivot_wider`, perform the operation and then `pivot_longer` back
  - If you need to modify some values according to some conditions, you can use either:
    - Logical indexing just like normal dataframes
    - `mutate(var = if_else(my_logical_vector, true_val, false_val))`
- `tidyr` helps to keep your data frame "clean":
  - A clean data frame should be where:
    - Each variable is in its own column;
    - Each observation is in its own row;
    - Each cell only have a single value
  - Pivotting (transposing), rectangling, nesting, splitting & combining, and dealing with missing values
- JAGS:
  - A package for bayesian modeling and MCMC
  - JAGS's syntax is similar to R, but it does **not** have if-statement!

## Modeling

- `contrasts`: setting contrast coding for categorical variables in the model

  - `contrast(df$var1) <- your_matrix`
  - Note: the row names of the matrix must be in the order of the levels (either set when you `factor()` you data, or alphabetical by default), otherwise it will be wrong!
  
- `lm`:

  - Linear fixed effect modeling

  - Example:

    ```R
    #define data
    df = data.frame(x=c(1, 3, 3, 4, 5, 5, 6, 8, 9, 12),
                    y=c(12, 14, 14, 13, 17, 19, 22, 26, 24, 22))
    
    #fit linear regression model using 'x' as predictor and 'y' as response variable
    model <- lm(y ~ x, data=df)
    ```

  - Summarizing results:

    - `summary(model)`
    - `plot(model)`: diagnostic plots (e.g. Q-Q plot)

  - Access results: `coefficients()`, `effects()`, `residuals()`, `fitted.values()` and so on

  - Make prediction: `predict(model, newdata = newdf)`

  - Note: for all model objects (including the ones below), do NOT save the object directly to files!

    - Because the model contains some `.Environment` attributes, so your entire workspace will be saved along with the model!
    - If you do so, the file size can be over 100MB while `object.size(mdl)` is only several MB
    - You can save the `coef(mdl)` and `VarCorr(mdl)` data frames instead, or `as.data.frame(mdl)` for `brms` models

- `glm`: fit linear fixed effect model with a link function
  - similar to `lm`, except an extra input `family` that describes the error distribution and link function
  - e.g., a logistric regression $y \sim \text{logit}(x_1 + x_2 + 1)$ can be fit with `glm(y ~ x1 + x2, data = df, family = binomial)`
    - the distribution of error should be binomial distribution
    - by default, the link funciton for binomial is logit

- `lmer`:
  - Linear mixed effect modeling, provided in the package `lme4`
  - Formula:
    - Random effect terms are represented as `design_matrices | group_variable` (usually needs to be in parentheses)
    - If there are more than one terms (apart from the intercept) in the `design_matrices`, their covariance will also be estimated
      - You can specify multiple independent random terms by replacing `|` with `||`, but all these terms must contain continuous rather than categorical data, otherwise you need to use `lmer_alt`
    - e.g., `y ~ x + (x | subjects)` represents:
      - a global intercept, a fixed effect for `x`, a random intercept (`+ 1` in the parentheses by default) for each subject, and the random deviation of the effect of `x` (i.e., its slope) across subject
      - $Y_{si} = \beta_0 + S_{0s} + (\beta_1 + S_{1s})X_i + \epsilon_{si}$, where $s, i$ are subject and stimulus index respectively, $\beta_0, \beta_1$ are fixed effects for intercept and stimulus respectively, $S_{0s}, S_{1s}$ are random deviations for intercept and stimulus effect on subject $s$ respectively, $X_i$ is the value for the $i$-th stimulus and $\epsilon_{si}$ is the error term
      - Note that the covariance between $S_{0s}$ and $S_{1s}$ across the group variable $s$ will also be estimated (i.e., they will be estimated together in a 2d Gaussian distribution). If you want to enforce a zero covariance, you need `y ~ x + (1 | subjects) + (0 + x | subjects)` instead.
    - e.g., `y ~ x1 + x2 + (x1 + x2 | subjects)` represents a global intercept, fixed effects for `x1` and `x2`, and random variance of (`x1`, `x2`, intercept and their covariance matrix) across subjects
  - Usage are the same as `lm`
  - Model will be fitted according to either maximum likelihood or REML criterion (indicator for estimation convergence)

- `brms`: Bayesian model fitting

  - Fitting model:

    ```R
    brm(formula, data, family = gaussian(link = "identity"), prior, ...,
      save_pars, ..., inits, chains, iter, warmup, thin, cores, threads, ...)
    ```

    - `formula`: almost identical as `lme4`, but you cannot use back quote for invalid variable names
    - `family`: the distribution of the response, e.g. `bernoulli(link = "logit")` for logistic regression
    - `link`: the link function between the right hand side and the left hand side
    - `save_pars`: an object generated by `save_pars()` controlling which parameters should be saved during MCMC
    - `prior`: an object defined by `set_prior()`
    - other MCMC parameters

    Return value is a fitted model of class `"brmsfit"`

  - For each variable `x` in the formula, `brm` will estimate its mean `b_x`, standard deviation `sd_x` and (if any) other parameters specific to its distribution

  - Prior can be set manually by

    ```R
    set_prior(prior, class = "b", coef, group, ..., lb, ub, ...)
    ```

    - `prior`: a distribution defined by `Stan` language
    - `class`: type of parameter using this prior, e.g., "b" for fixed effect and "sd" for standard deviation
    - `coef`: name of the parameter using this prior
    - `group`: name of the associated grouping factor
    - `lb`, `ub`: lower and upper bounds of the estimation

    Or you can use the default priors decided by `brms`, which can be viewed by `get_prior(formula, data, family, ...)`

  - Get results from the fitted model:

    - `fixef()`, `ranef()` for effects
    - `VarCorr()` for variances and covariances (e.g., "random" effect, residuals, etc.)
    - `fitted()` for the posterior distribution of the mean (expected) response
    - `predict()` for the posterior distribution of response (including the sampling variance)
    - `hypothesis()` for the posterior distribution of a function of the parameters (e.g., the difference)
      - `"xxx = 0"` for two sided test and `"xx > xxx"` for one sided test
      - For random effects, set `class = NULL` and use the full names of the term (e.g., `"sd_groupVar__rndTerm"`, see `names(as.data.frame(mdl))`) for testing
    - `conditional_effects()` for the effected of one predictor conditioned on the others
    - `bayes_R2()` for the Bayesian R^2^
    - `loo()` for model comparison

  - Note: the `Est.Error`  term in the outputs is the estimated standard deviation of this term

    - i.e., the standard deviation with n-1 dof for the samples, with 4000 samples in default
    - "Standard error" is not a bayesian concept, since the posterior of the estimated term already tells you everything about the uncertainty of the estimation
    - "Standard error" is just the posterior standard deviation

- `anova`: Analysis of variance or deviance

  - `anova(model)` computes the ANOVA table for a linear model (e.g., the result of `lm` or `lmer`)
  - `anova(model1, model2, model3, ...)` computes the likelihood ratio test of model1/model2, model2/model3, etc.
  - Note: type III sum-of-squares is standard, but the default `anova()` and `aov` function in R uses type I. You need to use `car:Anova()` to be able to specify the SS type.