# **Selected Heuristic Features (Fully Described)**

## **1️⃣ Material & King Safety**

### **1. material_diff_students**

Measures the material balance of student pieces.
Defined as the number of the current player’s students on the board minus the number of the opponent’s students on the board.

---

### **2. my_master_alive**

Indicator for whether the current player’s master is still present on the board.
Takes the value **1** if the master is present and **0** otherwise.

---

### **3. opp_master_alive**

Indicator for whether the opponent’s master is still present on the board.
Takes the value **1** if the opponent’s master is present and **0** otherwise.

---

### **4. master_safety_balance**

Measures the relative safety of the two masters.
Defined as the number of the current player’s available moves that can capture the opponent’s master **minus** the number of the opponent’s available moves that can capture the current player’s master.

---

## **2️⃣ Mobility & Threats**

### **6. legal_moves_diff**

Measures which side currently has greater tactical flexibility.
Defined as the number of legal moves available to the current player minus the number of legal moves available to the opponent.

---

### **7. capture_moves_diff**

Measures relative attacking potential.
Defined as the number of legal capture moves available to the current player minus the number of legal capture moves available to the opponent.

---

## **3️⃣ Positional / Progress**

### **9. master_temple_distance_diff**

Measures relative progress of the masters toward their respective victory temples.
Defined as the opponent master’s distance to the current player’s temple **minus** the current player’s master’s distance to the opponent’s temple.

---

### **10. student_progress_diff**

Measures which side’s student pieces are more advanced toward victory.
Defined as the opponent’s average student distance to the current player’s temple **minus** the current player’s average student distance to the opponent’s temple.

---

### **11. central_control_diff**

Measures which side controls the central region of the board.
Defined as the number of the current player’s pieces located in the 3×3 central zone of the board minus the number of the opponent’s pieces located in the same region.

---

## **4️⃣ Card-Related Features**

### **14. card_mobility_diff**

Measures overall mobility granted by the card sets.
Defined as the total number of legal moves enabled by the current player’s two cards minus the total number of legal moves enabled by the opponent’s two cards.

---

### **15. master_escape_options**

Measures how much mobility the current player’s master has using the current set of cards.
Defined as the number of legal moves available to the current player that involve moving the master.
