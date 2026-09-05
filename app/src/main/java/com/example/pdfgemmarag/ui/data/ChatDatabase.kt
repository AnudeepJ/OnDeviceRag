package com.example.pdfgemmarag.ui.data

import android.content.Context
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey val hash: String,
    val name: String,
    val pageCount: Int,
    val chunkCount: Int,
    val script: String,
    val indexedAt: Long,
    /** INDEXING, READY, FAILED */
    val status: String,
    val error: String? = null,
    val pdfPath: String,
)

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(index = true) val docHash: String,
    /** "user" or "model" */
    val role: String,
    val text: String,
    /** Comma-separated chunk ids ("<hash>:<index>") that supported the answer. */
    val citationIds: String = "",
    val createdAt: Long = System.currentTimeMillis(),
    val backend: String = "",
    val tokensPerSecond: Double = 0.0,
    val cancelled: Boolean = false,
)

@Dao
interface DocumentDao {
    @Query("SELECT * FROM documents ORDER BY indexedAt DESC")
    fun observeAll(): Flow<List<DocumentEntity>>

    @Query("SELECT * FROM documents WHERE hash = :hash")
    suspend fun get(hash: String): DocumentEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(doc: DocumentEntity)

    @Update
    suspend fun update(doc: DocumentEntity)

    @Delete
    suspend fun delete(doc: DocumentEntity)
}

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE docHash = :docHash ORDER BY createdAt ASC, id ASC")
    fun observe(docHash: String): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE docHash = :docHash ORDER BY createdAt DESC, id DESC LIMIT :limit")
    suspend fun latest(docHash: String, limit: Int): List<MessageEntity>

    @Insert
    suspend fun insert(message: MessageEntity): Long

    @Query("DELETE FROM messages WHERE docHash = :docHash")
    suspend fun deleteForDocument(docHash: String)
}

@Database(entities = [DocumentEntity::class, MessageEntity::class], version = 1, exportSchema = true)
abstract class ChatDatabase : RoomDatabase() {
    abstract fun documents(): DocumentDao
    abstract fun messages(): MessageDao

    companion object {
        fun create(context: Context): ChatDatabase =
            Room.databaseBuilder(context, ChatDatabase::class.java, "rag_chat.db")
                .fallbackToDestructiveMigration(dropAllTables = true)
                .build()
    }
}
