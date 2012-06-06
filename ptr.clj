(defn revise-weights [weights  [input error] learning-rate]
  (map (fn [w xi] (+ w (* learning-rate error xi))) weights input))

(defn classify [weights input]
  (if (> (reduce + (map * weights input)) 0.5) 1 0))

(defn check-weights [weights training-set]
  (reduce
    (fn [m [i r]]
      (let [e (- r (classify weights i))]
        (if (not= e 0) (assoc m i e) m)))
    {}
    training-set))

(defn perceptron-training-rule [training-set weights learning-rate]
  (let [errored (check-weights weights training-set)]
    (if (seq errored)
      (perceptron-training-rule
        training-set
        (reduce #(revise-weights %1 %2 learning-rate) weights errored)
        learning-rate)
      weights)))