; Naive bayesian classifier. References - ML Mitchell ch 6, ML in action ch 4
; Author:  Jitesh Sejwal

(ns user
  (:use [clojure.string :only [split trim lower-case]]))
(defn occurence-count [vocab input]
  "Returns a map of each word in vocab to number of occurences of that word in the input vector." 
  (reduce
    #(assoc %1 %2 (count (filter (fn [x] (= x %2)) input)))
    {}
    vocab))

(defn create-vocab [dataset]
  "Returns the vocabulary for a sequence of examples."
  (reduce (fn [s m] (reduce conj s m)) #{} dataset))

(defn learn-naive-bayes-text
  ([examples results]
    (assert (= (count examples) (count results)))
    (into
      {}
      (map
        (fn [y] [y (learn-naive-bayes-text examples results y)])
        (distinct results))))
  ([examples results val]    
    (frequencies
      (mapcat (fn [d r] (if (= val r) d)) examples results))))

(defn get-0 [m k]
  (if-let [ret (get m k)] ret 0))

(defn cond-prob [learned-text word result denom]
  (/ (inc (get-0 (get learned-text result) word))
     denom))

(defn get-probabilities [examples result input result-set]
  (let [l-text (learn-naive-bayes-text examples result)
        vocab-count (count (create-vocab examples))]
    (map
      (fn [v]
        (let [f-vals (map #(if (= %2 v) (count %1) 0) examples result)
              n (reduce + f-vals)
              Pv (/ (count (filter #(> % 0) f-vals)) (count result))]
          [v
           (reduce
             #(+ %1 (Math/log (cond-prob l-text %2 v (+ n vocab-count))))
             (Math/log Pv)
             input)]))
      result-set)))

(defn classify [examples result input result-set]
  (first
    (reduce
      #(max-key second %1 %2)
      (get-probabilities examples result input result-set))))


;================================================
;https://github.com/pbharrin/machinelearninginaction/blob/master/Ch04/email.zip
;================================================
(defn from-file
  ([file sep] (-> file slurp trim lower-case (#'clojure.string/split sep) )))
(def ham1 (map #(from-file (str "G:\\email\\ham\\" % ".txt") #"\r\n| ") (range 1 26)))
(def ham (map (fn [v] (filter #(.matches % "\\w+") v)) ham1))
(def spam1 (map #(from-file (str "G:\\email\\spam\\" % ".txt") #"\r\n| ") (range 1 26)))
(def spam (map (fn [v] (filter #(.matches % "\\w+") v)) spam1))
(map
  #(classify
    (concat spam ham)
    (concat (repeat 25 "spam") (repeat 25 "ham"))
    %
    ["ham" "spam"])
  ham)

