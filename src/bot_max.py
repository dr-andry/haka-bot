import asyncio
import logging
import json
from typing import Dict, Any, List

from maxapi import Bot, Dispatcher
from maxapi.filters import F
from maxapi.types import (
    MessageCreated, MessageCallback, CommandStart,
    CallbackButton, ButtonsPayload
)
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder

# Импортируем функцию для рекомендаций
from rank_events import get_top_k_for_person

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os

token = os.environ.get("TOKEN")
print(token)

bot = Bot(token)
dp = Dispatcher()

# Хранилище состояний пользователей
user_states: Dict[int, Dict[str, Any]] = {}

# ========== FSM STATES ==========
class UserState:
    SPECIALITY = "speciality"
    INTERESTS = "interests"
    COURSE = "course"
    DONE = "done"

def get_user_id(event) -> int:
    """Безопасное получение user_id из разных типов событий"""
    try:
        # Для MessageCreated событий
        if hasattr(event, 'message') and hasattr(event.message, 'user_id'):
            return event.message.user_id
        # Для MessageCallback событий  
        elif hasattr(event, 'user_id'):
            return event.user_id
        # Для других событий с from_user
        elif hasattr(event, 'from_user') and hasattr(event.from_user, 'user_id'):
            return event.from_user.user_id
        else:
            # Если не можем получить user_id, используем chat_id как fallback
            return get_chat_id(event)
    except Exception as e:
        logger.error(f"Error getting user_id: {e}")
        return 0

def get_chat_id(event) -> int:
    """Безопасное получение chat_id из разных типов событий"""
    try:
        if hasattr(event, 'chat_id') and event.chat_id:
            return event.chat_id
        elif hasattr(event, 'message') and hasattr(event.message, 'chat_id'):
            return event.message.chat_id
        elif hasattr(event, 'chat') and hasattr(event.chat, 'chat_id'):
            return event.chat.chat_id
        else:
            logger.error("Cannot determine chat_id")
            return 0
    except Exception as e:
        logger.error(f"Error getting chat_id: {e}")
        return 0

def get_callback_payload(callback) -> str:
    """Безопасное получение payload из callback"""
    try:
        # В MAX API callback - это объект с атрибутом payload
        if hasattr(callback, 'callback') and hasattr(callback.callback, 'payload'):
            return callback.callback.payload
        # Альтернативные варианты на всякий случай
        elif hasattr(callback, 'payload') and callback.payload:
            return callback.payload
        elif hasattr(callback, 'data') and callback.data:
            return callback.data
        else:
            return ""
    except Exception as e:
        logger.error(f"Error getting callback payload: {e}")
        return ""

# ========== START COMMAND ==========
@dp.message_created(CommandStart())
async def start_command(event: MessageCreated):
    """Обработка команды /start - показывает начальные кнопки"""
    try:
        user_id = get_user_id(event)
        chat_id = get_chat_id(event)
        
        logger.info(f"Start command received from user_id: {user_id}, chat_id: {chat_id}")
        
        # Сбрасываем состояние пользователя
        user_states[user_id] = {"step": None}
        
        # Создаем клавиатуру с кнопками
        builder = InlineKeyboardBuilder()
        builder.row(
            CallbackButton(text="🚀 Начать", payload="start_fsm"),
            CallbackButton(text="❌ Закончить", payload="cancel_fsm")
        )
        
        await event.message.answer(
            text="👋 Привет! Я бот, который подбирает мероприятия для студентов.\n\n"
                 "Нажми кнопку ниже, чтобы начать или закончить подбор.",
            attachments=[builder.as_markup()]
        )
        
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        try:
            await event.message.answer("❌ Произошла ошибка. Попробуйте еще раз.")
        except:
            pass

# ========== CANCEL COMMAND ==========
@dp.message_created(F.message.body.text == "/cancel")
async def cancel_command(event: MessageCreated):
    """Обработка команды отмены"""
    try:
        user_id = get_user_id(event)
        user_states[user_id] = {"step": None}
        
        await event.message.answer("❌ Отменено. Нажми /start чтобы начать снова.")
    except Exception as e:
        logger.error(f"Error in cancel_command: {e}")

# ========== CALLBACK HANDLERS ==========
@dp.message_callback()
async def handle_callbacks(callback: MessageCallback):
    """Обработка нажатий на кнопки"""
    try:
        user_id = get_user_id(callback)
        payload = get_callback_payload(callback)
        
        logger.info(f"Callback received from user_id: {user_id}, payload: {payload}")
        
        if payload == "start_fsm":
            # Начинаем FSM процесс
            user_states[user_id] = {"step": UserState.SPECIALITY}
            await callback.message.answer("🎯 Введите вашу специальность:")
            
        elif payload == "cancel_fsm":
            # Отменяем FSM процесс
            user_states[user_id] = {"step": None}
            await callback.message.answer("❌ Отменено. Нажми /start чтобы начать снова.")
        else:
            logger.warning(f"Unknown payload received: {payload}")
            await callback.message.answer("❌ Неизвестная команда. Нажми /start чтобы начать.")
        
        # Подтверждаем обработку callback
        if hasattr(callback, 'answer'):
            await callback.answer()
            
    except Exception as e:
        logger.error(f"Error in handle_callbacks: {e}")
        try:
            await callback.message.answer("❌ Произошла ошибка. Попробуйте еще раз.")
        except:
            pass

# ========== FSM MESSAGE HANDLERS ==========
@dp.message_created(F.message.body.text)
async def handle_text_messages(event: MessageCreated):
    """Обработка текстовых сообщений для FSM"""
    try:
        user_id = get_user_id(event)
        text = event.message.body.text.strip()
        
        # Игнорируем команды (они обрабатываются отдельно)
        if text.startswith('/'):
            return
            
        # Если пользователь не в FSM, игнорируем обычные сообщения
        if user_id not in user_states or not user_states[user_id].get("step"):
            # Можно ответить, что нужно начать с /start
            await event.message.answer("💡 Нажмите /start чтобы начать подбор мероприятий.")
            return
        
        current_state = user_states[user_id]["step"]
        
        if current_state == UserState.SPECIALITY:
            await handle_speciality_input(event, text, user_id)
            
        elif current_state == UserState.INTERESTS:
            await handle_interests_input(event, text, user_id)
            
        elif current_state == UserState.COURSE:
            await handle_course_input(event, text, user_id)
            
    except Exception as e:
        logger.error(f"Error in handle_text_messages: {e}")
        try:
            await event.message.answer("❌ Произошла ошибка. Попробуйте еще раз.")
        except:
            pass

async def handle_speciality_input(event: MessageCreated, text: str, user_id: int):
    """Обработка ввода специальности"""
    try:
        user_states[user_id]["speciality"] = text
        user_states[user_id]["step"] = UserState.INTERESTS
        
        await event.message.answer(
            "📚 Введите ваши интересы через запятую:\n\n"
            "Пример: аналитика, экономика, AI-инструменты, программирование"
        )
    except Exception as e:
        logger.error(f"Error in handle_speciality_input: {e}")
        raise

async def handle_interests_input(event: MessageCreated, text: str, user_id: int):
    """Обработка ввода интересов"""
    try:
        # Разделяем интересы по запятым и очищаем от лишних пробелов
        interests = [interest.strip() for interest in text.split(",") if interest.strip()]
        
        if not interests:
            await event.message.answer("❌ Пожалуйста, введите хотя бы один интерес. Попробуйте снова:")
            return
        
        user_states[user_id]["interests"] = interests
        user_states[user_id]["step"] = UserState.COURSE
        
        await event.message.answer("🎓 Введите ваш курс (1-4):")
    except Exception as e:
        logger.error(f"Error in handle_interests_input: {e}")
        raise

async def handle_course_input(event: MessageCreated, text: str, user_id: int):
    """Обработка ввода курса и вывод рекомендаций"""
    try:
        course = int(text.strip())
        if course < 1 or course > 4:
            raise ValueError
    except ValueError:
        await event.message.answer("❌ Пожалуйста, введите число от 1 до 4:")
        return
    
    try:
        user_states[user_id]["course"] = course
        user_states[user_id]["step"] = UserState.DONE
        
        # Показываем что ищем мероприятия
        search_message = await event.message.answer("🔎 Ищу лучшие мероприятия...")
        
        # Формируем данные пользователя
        person = {
            'speciality': user_states[user_id]["speciality"],
            'interests': user_states[user_id]["interests"],
            'course': course
        }
        
        logger.info(f"Searching events for user {user_id}: {person}")
        
        # Получаем рекомендации
        recommendations = get_top_k_for_person(person, k=5)
        
        # Удаляем сообщение "Ищу..."
        try:
            if hasattr(search_message, 'message_id'):
                await bot.delete_message(search_message.message_id)
        except Exception as e:
            logger.warning(f"Could not delete search message: {e}")
        
        if not recommendations:
            await event.message.answer(
                "😢 К сожалению, по вашим критериям ничего не найдено.\n\n"
                "Попробуйте изменить специальность или интересы."
            )
        else:
            # Формируем сообщение с рекомендациями
            message_text = "✨ **Топ мероприятий для вас:**\n\n"
            
            for i, event_data in enumerate(recommendations, 1):
                # Обрезаем описание если слишком длинное
                description = event_data['description']
                if len(description) > 150:
                    description = description[:147] + "..."
                
                message_text += (
                    f"**{i}. {event_data['title']}**\n"
                    f"   {description}\n"
                    f"   🔗 {event_data['url']}\n"
                    f"   📊 Релевантность: {event_data['score']:.2f}\n\n"
                )
            
            await event.message.answer(message_text)
            
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {e}")
        await event.message.answer(
            "⚠️ Произошла ошибка при поиске мероприятий. "
            "Попробуйте позже или обратитесь к администратору."
        )
    
    finally:
        # Сбрасываем состояние и предлагаем начать заново
        user_states[user_id] = {"step": None}
        
        # Показываем кнопки для повторного запуска
        builder = InlineKeyboardBuilder()
        builder.row(
            CallbackButton(text="🔄 Начать заново", payload="start_fsm")
        )
        
        await event.message.answer(
            "Готово! Хотите попробовать с другими параметрами?",
            attachments=[builder.as_markup()]
        )

# ========== BOT STARTUP ==========
async def main():
    """Запуск бота"""
    try:
        logger.info("Запуск бота для рекомендаций мероприятий...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.close()

if __name__ == '__main__':
    asyncio.run(main())