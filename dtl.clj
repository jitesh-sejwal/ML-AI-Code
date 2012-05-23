;; Decision Tree Learning Impl - ID3
;; 
;; References: Machine Learning - Mitchell, Macthine Learning in Action
;;
;; Author: Jitesh Sejwal
(ns user
  (:use [clojure.contrib.def])
  (:use [clojure.set :only [difference]]))

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

(defn subdataset-attr-val [dataset attr val]
  "Returns the sub-dataset of dataset such that each element's
   attribute attr has value val."
  (filter #(= (get % attr) val) dataset))

(defn partition-dataset [dataset attr]
  "Partitions dataset around the attribute attr. Each element is a 
   sub-dataset of dataset such that all elements of that dataset have
   the same value for the attribute attr."
  (map
    #(subdataset-attr-val dataset attr %)
    (distinct (map attr dataset))))

(defn info-gain [dataset attr]
  (-
    (entropy dataset)
    (reduce
      #(+ %1 (* (/ (count %2) (count dataset)) (entropy %2)))
      0
      (partition-dataset dataset attr))))

(defn max-info-gain-attr [dataset attrs]
  (reduce
    (fn [x y] (max-key #(info-gain dataset %) x y))
    attrs))

(defn result-with-max-frequency [dataset]
  (first
    (reduce
      #(if (> (second %1) (second %2)) %1 %2)
      (seq (frequencies (map *result* dataset))))))

(defn id3
  ([dataset]
    (id3
      dataset
      (set (remove #(= *result* %) (keys (first dataset))))))
  ([dataset attrs]
    (if (empty? attrs)
      {*result* (result-with-max-frequency dataset)}
      (let [max-attr (max-info-gain-attr dataset attrs)]
        (assoc {} max-attr (id3 dataset max-attr attrs)))))
  ([dataset target-attr attrs]
    (reduce
      (fn [ret value]
        (let [sub-attr-val (subdataset-attr-val dataset target-attr value)]
          (if (every? #(= (*result* (first sub-attr-val)) (*result* %)) sub-attr-val)
            (assoc ret value {*result* (*result* (first sub-attr-val))})
            (assoc
              ret
              value
              (id3 sub-attr-val (difference attrs #{target-attr}))))))
      {}
      (distinct (map target-attr dataset)))))

(defn classify0 [tree input]
  (let [key (first (keys tree))]
    (if (= *result* key)
      (*result* tree)
      (if-let [valu (key input)]
        (if-let [t (get tree key)]
          (if-let [t2  (get t valu)]
            ('classify0 t2 input)))))))

(defn unit-tests []
  (do (let [tennis [{:outlook "Sunny",
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
    (assert (= :outlook (max-info-gain-attr tennis #{:outlook :wind :temperature :humidity})))
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
        (id3 tennis #{:outlook :wind :temperature :humidity})))
    :passed)
  (let [lenses [{:age"young",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "young",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "soft"}
                {:age "young",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "young",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "hard"}
                {:age "young",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "young",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "soft"}
                {:age "young",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "young",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "hard"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "soft"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "hard"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "soft"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "pre-presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "myope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "hard"}
                {:age "presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "no",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "no",
                 :result "soft"}
                {:age "presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "reduced",
                 :astigmatic "yes",
                 :result "no-lenses"}
                {:age "presbyopic",
                 :spectacle-prescription "hypermetrope",
                 :tear-production-rate "normal",
                 :astigmatic "yes",
                 :result "no-lenses"}]]
    (assert
      (=
        {:tear-production-rate
         {"normal"
          {:astigmatic
           {"yes"
            {:spectacle-prescription
             {"hypermetrope"
              {:age
               {"presbyopic" {:result "no-lenses"},
                "pre-presbyopic" {:result "no-lenses"},
                "young" {:result "hard"}}},
              "myope" {:result "hard"}}},
            "no"
            {:age
             {"presbyopic"
              {:spectacle-prescription
               {"hypermetrope" {:result "soft"},
                "myope" {:result "no-lenses"}}},
              "pre-presbyopic" {:result "soft"},
              "young" {:result "soft"}}}}},
          "reduced" {:result "no-lenses"}}}
        (id3 lenses #{:age :astigmatic :spectacle-prescription :tear-production-rate})))
    :passed))
  :all-passed)