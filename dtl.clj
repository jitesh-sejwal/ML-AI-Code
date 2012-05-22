;; Decision Tree Learning Impl - ID3
;; 
;; References: Machine Learning - Mitchell, Macthine Learning in Action
;;
;; Author: Jitesh Sejwal
(use 'clojure.contrib.def)

(defn lg [x]
  (/ (Math/log x) (Math/log 2)))

(defvar ^:dynamic *result* :result)

(defn entropy [u]
  (reduce
    +
    0
    (map
      #(let [p (/ % (count u))] (* -1 p (lg p)))
      (vals (frequencies (map *result* u))))))

(defn partition-dataset
  ([dataset attr]
    (map
      #(partition-dataset dataset attr %)
      (distinct (map attr dataset))))
  ([dataset attr value]
    (for [v dataset :when (= (attr v) value)] (dissoc v attr))))

(defn info-gain [dataset attr]
  (-
    (entropy dataset)
    (reduce
      #(+ %1 (* (/ (count %2) (count dataset)) (entropy %2)))
      0
      (partition-dataset dataset attr))))

(defn max-info-gain-attr [dataset]
  (reduce
    (fn [x y] (max-key #(info-gain dataset %) x y))
    (remove #(= *result* %) (keys (first dataset)))))

(defn empty-attrs? [dataset]
  (if-let [k (keys dataset)]
    (and (= (count k) 1) (= *result* (first k)))))

(defn id3
  ([dataset]
    (if (empty-attrs? dataset)
      {*result*
       (first
         (reduce
           #(if (> (second %1) (second %2)) %1 %2)
           (seq (frequencies (map *result* dataset)))))}
      (let [max-attr (max-info-gain-attr dataset)]
        (assoc {} max-attr (#'make-tree dataset max-attr)))))
  ([dataset target-attr]
    (reduce
      (fn [ret value]
        (let [part (partition-dataset dataset target-attr value)]
          (if (every? #(= (*result* (first part)) (*result* %)) part)
            (assoc ret value {*result* (*result* (first part))})
            (assoc ret value (make-tree part)))))
      {}
      (distinct (map target-attr dataset)))))

(defn classify0 [tree input]
  (let [key (first (keys tree))]
    (if (= *result* key)
      (*result* tree)
      (if-let [valu (key input)]
        (if-let [t (get tree key)]
          (if-let [t2  (get t valu)]
            (#'classify0 t2 input)))))))

(defn unit-tests []
  (let [tennis [{:outlook "Sunny",
                 :temperature "Hot",
                 :humidity "High",
                 :wind "Weak",
                 :result "No"}
                {:outlook "Sunny",
                 :temperature "Hot",
                 :humidity "High",
                 :wind "Strong",
                 :result "No"}
                {:outlook "Overcast",
                 :temperature "Hot",
                 :humidity "High",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Rain",
                 :temperature "Mild",
                 :humidity "High",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Rain",
                 :temperature "Cool",
                 :humidity "Normal",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Rain",
                 :temperature "Cool",
                 :humidity "Normal",
                 :wind "Strong",
                 :result "No"}
                {:outlook "Overcast",
                 :temperature "Cool",
                 :humidity "Normal",
                 :wind "Strong",
                 :result "Yes"}
                {:outlook "Sunny",
                 :temperature "Mild",
                 :humidity "High",
                 :wind "Weak",
                 :result "No"}
                {:outlook "Sunny",
                 :temperature "Cool",
                 :humidity "Normal",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Rain",
                 :temperature "Mild",
                 :humidity "Normal",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Sunny",
                 :temperature "Mild",
                 :humidity "Normal",
                 :wind "Strong",
                 :result "Yes"}
                {:outlook "Overcast",
                 :temperature "Mild",
                 :humidity "High",
                 :wind "Strong",
                 :result "Yes"}
                {:outlook "Overcast",
                 :temperature "Hot",
                 :humidity "Normal",
                 :wind "Weak",
                 :result "Yes"}
                {:outlook "Rain",
                 :temperature "Mild",
                 :humidity "High",
                 :wind "Strong",
                 :result "No"}]]
    (assert (= :outlook (max-info-gain-attr tennis)))
    (assert (< (- (entropy tennis) 0.94) 5.0E-4))
    (assert
      (=
        {:outlook
         {"Rain"
          {:wind {"Strong" {:result "No"}, "Weak" {:result "Yes"}}},
          "Overcast" {:result "Yes"},
          "Sunny"
          {:humidity
           {"Normal" {:result "Yes"}, "High" {:result "No"}}}}}
        (id3 tennis)))
    :passed))


